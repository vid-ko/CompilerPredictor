import predictor.CompilerPredictor;
import ai.onnxruntime.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Benchmarking {

    private static double onnxExecutionTime;
    private static double executionTime;

    private static final String FILENAME_TEST_DATA = "testDataResNet.csv";
    private static final int INPUT_SIZE = 147;
    private static final String FILENAME_MODEL = "modelResNet.onnx";

    public static void main(String[] args){

        System.out.println("Starting... ");



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
        floats = new float[records.get(2).size()];
        for (int i = 0; i < records.get(2).size(); i++){
            floats[i] = Float.parseFloat(records.get(2).get(i));
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
