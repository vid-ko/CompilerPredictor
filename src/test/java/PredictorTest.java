import activation.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import predictor.NeuralNetwork;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PredictorTest {

    public static void main(String[] args){

        System.out.println("Starting... ");

        NeuralNetwork ann = new NeuralNetwork("model.onnx");

        // TODO: read input data
        float[] input = Arrays.copyOfRange(readTestDataFromCSV("src/test/java/test_data.csv"), 0, 120);
        var result = ann.predict(input);
        
        for(var r : result){
            System.out.println(r);
        }




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


}
