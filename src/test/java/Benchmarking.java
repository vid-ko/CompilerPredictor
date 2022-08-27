import predictor.CompilerPredictor;
import predictor.OnnxPredictor;
import predictor.Predictor;
import predictor.TribuoPredictor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Benchmarking {

    private static double tribuoExecutionTime;
    private static double onnxExecutionTime;
    private static double executionTime;

    private static final String FILENAME_TEST_DATA = "testDataResNet2.csv";
    private static final int INPUT_SIZE = 147;
    private static final String FILENAME_MODEL = "modelResNet.onnx";

    public static void main(String[] args){

        System.out.println("Starting... ");

        // read input data
        float[] input = Arrays.copyOfRange(readTestDataFromCSV("src/test/java/" + FILENAME_TEST_DATA), 0, INPUT_SIZE);


        var tribuoResult = predictUsingTribuo(input);
        var onnxResult = predictUsingONNXRuntime(input);
        var result = predictUsingCompilerPredictor(input);


        System.out.println("CP                        ONNX                       Tribuo");
        for(int i = 0; i < 5; i++){
            System.out.printf("%.8f               %.8f                %.8f\n", result[i],onnxResult[i],tribuoResult[i]);
        }
        System.out.println("--------------------------------------------------------------------------------");
        System.out.println("");
        System.out.println("Average value difference: " + averageDifference(result, onnxResult));
        System.out.println("CP execution time: " + executionTime +" seconds");
        System.out.println("ONNX execution time: " + onnxExecutionTime +" seconds");
        System.out.println("Tribuo execution time: " + tribuoExecutionTime +" seconds");


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
        Predictor predictor = new OnnxPredictor(FILENAME_MODEL);

        long startingTime = System.nanoTime();
        float[] output = predictor.predict(testData);
        long finishTime = System.nanoTime();
        tribuoExecutionTime = (double)(finishTime - startingTime)/1000000000;

        return output;
    }

    private static float[] predictUsingTribuo(float[] testData){
        Predictor predictor = new TribuoPredictor(FILENAME_MODEL, INPUT_SIZE, 5);

        long startingTime = System.nanoTime();
        float[] output = predictor.predict(testData);
        long finishTime = System.nanoTime();
        onnxExecutionTime = (double)(finishTime - startingTime)/1000000000;

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
