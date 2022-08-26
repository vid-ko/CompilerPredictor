import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.data.csv.CSVIterator;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.interop.onnx.*;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import predictor.CompilerPredictor;
import ai.onnxruntime.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;

public class Benchmarking {

    private static double onnxExecutionTime;
    private static double executionTime;

    private static final String FILENAME_TEST_DATA = "testDataResNet2.csv";
    private static final int INPUT_SIZE = 147;
    private static final String FILENAME_MODEL = "modelResNet.onnx";

    public static void main(String[] args){

        System.out.println("Starting... ");

        //predictUsingTribuo(null);




        // read input data
        float[] input = Arrays.copyOfRange(readTestDataFromCSV("src/test/java/" + FILENAME_TEST_DATA), 1, INPUT_SIZE+1);




        var result = predictUsingCompilerPredictor(input);
        var onnxResult = predictUsingONNXRuntime(input);

        System.out.println("CP                        ONNX");
        for(int i = 0; i < 5; i++){
            System.out.printf("%.8f               %.8f\n", result[i],onnxResult[i]);
        }
        System.out.println("------------------------------------");
        System.out.println("");
        System.out.println("Average value difference: " + averageDifference(result, onnxResult));
        System.out.println("CP execution time: " + executionTime +" seconds");
        System.out.println("ONNX execution time: " + onnxExecutionTime +" seconds");




    }

    private static float[] predictUsingCompilerPredictor(float[] input){
        CompilerPredictor cp = new CompilerPredictor(FILENAME_MODEL);

        long startingTime = System.nanoTime();
        float[] result = cp.predict(input);
        long finishTime = System.nanoTime();
        executionTime = (double)(finishTime - startingTime) / 1000000000;
        return result;
    }

    private static float[] predictUsingONNXRuntime(float[] testData){

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        float[] output = new float[5];
        try {

            float[][] features = new float[1][INPUT_SIZE];
            for (int i = 0; i < INPUT_SIZE; i++){
                features[0][i] = testData[i];
            }

            OnnxTensor tensor = OnnxTensor.createTensor(env, features);
            Map<String, OnnxTensor> input = new HashMap<>();


            OrtSession session = env.createSession(FILENAME_MODEL, new OrtSession.SessionOptions());
            input.put(session.getInputNames().toArray()[0].toString(), tensor);


            long startingTime = System.nanoTime();
            try (var result = session.run(input)) {
                float[][] res = (float[][])result.get(0).getValue();
                output = res[0];
            }
            long finishTime = System.nanoTime();
            onnxExecutionTime = (double)(finishTime - startingTime)/1000000000;


            env.close();

        } catch (OrtException e) {
            e.printStackTrace();
        }

        return output;
    }

    private static float[] predictUsingTribuo(float[] testData){


        RegressionFactory regressionFactory = new RegressionFactory();



        Map<Regressor,Integer> ptOutMapping = new HashMap<>();
        for (int i = 0; i < 5; i++) {
            ptOutMapping.put(new Regressor("y"+i,0 ), i);
        }
        //ptOutMapping.put(new Regressor("255", 0), 0);
        String[] headers = new String[INPUT_SIZE];
        for (int i = 0; i < headers.length; i++){
            headers[i] = "x";
        }
        List<Prediction<Regressor>> result = null;
        var ortEnv = OrtEnvironment.getEnvironment();
        var sessionOpts = new OrtSession.SessionOptions();

        var denseTransformer = new DenseTransformer();

        try {
            //OrtSession session = ortEnv.createSession(FILENAME_MODEL, new OrtSession.SessionOptions());

            Map<String, Integer> ptFeatMapping = new HashMap<>();
            for (int i = 0; i < INPUT_SIZE; i++) {
                ptFeatMapping.put("x"+i, i);
            }
            //ptFeatMapping.put("input.1",0);



            var model = ONNXExternalModel.createOnnxModel(regressionFactory, ptFeatMapping, ptOutMapping,
                    denseTransformer, new RegressorTransformer(), sessionOpts, Paths.get(FILENAME_MODEL), "input.1");
            var testFilePath = Paths.get("src/test/java/" +FILENAME_TEST_DATA);

            var csvLoader = new CSVLoader<>(regressionFactory);
            var dataSource = csvLoader.loadDataSource(testFilePath,"net_output_0");
            var splitter = new TrainTestSplitter<>(dataSource, 0.0f, 0L);
            Dataset<Regressor> evalData = new MutableDataset<>(splitter.getTest());

            //var dataSource = new CSVLoader<>(regressionFactory).loadDataSource( testFilePath,"x", headers);


            //var s = regressionFactory.getEvaluator().evaluate(model, evalData).toString();
            //System.out.println(s);
            result = model.predict(evalData);

            var z = 5;
        } catch (OrtException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


        return null;
    }

    private static float[] readTestDataFromCSV(String filename){
        float[] floats = null;
        List<List<String>> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                records.add(Arrays.asList(values));
            }
        } catch (IOException e){
            e.printStackTrace();
        }
        floats = new float[records.get(1).size()];
        for (int i = 0; i < records.get(1).size(); i++){
            floats[i] = Float.parseFloat(records.get(1).get(i));
        }
        return floats;
    }


    private static float averageDifference(float[] values1, float[] values2){
        float sum = 0;
        for (int i = 0; i < values1.length; i++){
            sum+= values1[i] - values2[i];
        }
        return sum/values1.length;
    }




}
