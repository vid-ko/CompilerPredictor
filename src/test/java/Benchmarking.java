import predictor.NeuralNetwork;
import ai.onnxruntime.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Benchmarking {

    private static double onnxExecutionTime;
    private static double executionTime;

    public static void main(String[] args){

        System.out.println("Starting... ");



        // read input data
        float[] input = Arrays.copyOfRange(readTestDataFromCSV("src/test/java/test_data.csv"), 1, 121);

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
        NeuralNetwork ann = new NeuralNetwork("model.onnx");

        long startingTime = System.nanoTime();
        float[] result = ann.predict(input);
        long finishTime = System.nanoTime();
        long duration = finishTime - startingTime;
        executionTime = (double)(finishTime - startingTime) / 1000000000;
        return result;
    }

    private static float[] predictUsingONNXRuntime(float[] testData){

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        float[] output = new float[5];
        try {

            float[][] features = new float[1][120];
            for (int i = 0; i < 120; i++){
                features[0][i] = testData[i];
            }

            OnnxTensor tensor = OnnxTensor.createTensor(env, features);
            Map<String, OnnxTensor> input = new HashMap<>();
            input.put("0", tensor);

            OrtSession session = env.createSession("model.onnx", new OrtSession.SessionOptions());



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
