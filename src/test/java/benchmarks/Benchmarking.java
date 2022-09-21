package benchmarks;

import org.nd4j.linalg.factory.Nd4j;
import predictor.CompilerPredictor;
import predictor.OnnxPredictor;
import predictor.Predictor;
import predictor.TribuoPredictor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Benchmarking {

    private static double tribuoExecutionTime;
    private static double onnxExecutionTime;
    private static double cpExecutionTime;

    private static final String FILENAME_TEST_DATA = "testDataResNet.csv";
    private static final int INPUT_SIZE = 147;
    private static final String INPUT_NAME = "input.1";
    private static final String FILENAME_MODEL1 = "modelSmall.onnx";
    private static final String FILENAME_MODEL2 = "modelLarge.onnx";
    private static final String FILENAME_MODEL3 = "modelResNet.onnx";


    public static void main(String[] args){

        System.out.println("Starting... ");

        var testData = readTestDataFromCSV("src/test/java/" + FILENAME_TEST_DATA,1);
        for(var input : testData) {

            var onnxResult = predictUsingONNXRuntime(input);
            var result = predictUsingCompilerPredictor(input);
            var tribuoResult = predictUsingTribuo(input);

            System.out.println("CP                        ONNX                       Tribuo");
            for (int i = 0; i < 5; i++) {
                System.out.printf("%.8f               %.8f                %.8f\n", result[i], onnxResult[i], tribuoResult[i]);
            }
            System.out.println("--------------------------------------------------------------------------------");
            System.out.println("");
            //System.out.println("Average value difference: " + averageDifference(result, onnxResult));
            System.out.println("CP execution time: " + cpExecutionTime + " ms");
            System.out.println("ONNX execution time: " + onnxExecutionTime + " ms");
            System.out.println("Tribuo execution time: " + tribuoExecutionTime + " ms");



        }
    }

    private static float[] predictUsingCompilerPredictor(float[]testData){
        CompilerPredictor cp = null;
        Nd4j.create(1,1);
        try {
            long startingTime = System.nanoTime();
            cp = new CompilerPredictor(FILENAME_MODEL1);
            long finishTime = System.nanoTime();
            cpExecutionTime += (double) (finishTime - startingTime) / 1000000;
        } catch (Exception e) {
            e.printStackTrace();
        }

        var result = cp.predict(testData);

        return result;
    }

    private static float[] predictUsingONNXRuntime(float[] testData){
        long startingTime = System.nanoTime();
        Predictor predictor = new OnnxPredictor(FILENAME_MODEL1);
        long finishTime = System.nanoTime();
        onnxExecutionTime += (double) (finishTime - startingTime) / 1000000;

        var output = predictor.predict(testData);

        return output;
    }

    private static float[] predictUsingTribuo(float[] testData){
        long startingTime = System.nanoTime();
        Predictor predictor = new TribuoPredictor(FILENAME_MODEL1,INPUT_NAME, INPUT_SIZE, 5);
        long finishTime = System.nanoTime();
        tribuoExecutionTime += (double) (finishTime - startingTime) / 1000000;

        var output = predictor.predict(testData);

        return output;
    }


    private static float[][] readTestDataFromCSV(String filename, int numRows) {
        float[][] floats;
        List<List<String>> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                records.add(Arrays.asList(values));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        floats = new float[numRows][INPUT_SIZE];
        for (int j = 0; j < numRows; j++) {
            for (int i = 0; i < records.get(j+1).size()-6; i++) {
                floats[j][i] = Float.parseFloat(records.get(j+1).get(i+1));
            }
        }
        return floats;
    }

}
