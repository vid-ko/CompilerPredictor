package benchmarks;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.BenchmarkParams;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
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
import java.util.concurrent.TimeUnit;

public class BenchmarkingJMH {

    private static final String FILENAME_TEST_DATA = "testDataResNet.csv";
    private static final int INPUT_SIZE = 147;
    private static final String INPUT_NAME = "input.1";
    private static final String FILENAME_MODEL = "modelResNet.onnx";


    public static void main(String[] args) throws Exception {
        Options opt = new OptionsBuilder()
                .include(BenchmarkingJMH.class.getSimpleName())
                .mode(Mode.AverageTime)
                .forks(1)
                .warmupIterations(0)
                .measurementIterations(1)
                .timeUnit(TimeUnit.MILLISECONDS)
                .build();

        new Runner(opt).run();
    }

    @State(Scope.Benchmark)
    public static class PredictorProvider{
        private Predictor cpPredictor;
        private Predictor onnxPredictor;
        private Predictor tribuoPredictor;

        private float[][] features;

        public PredictorProvider(){

        }


        @Setup
        public void setup(BenchmarkParams params){
            features =readTestDataFromCSV("src/test/java/" + FILENAME_TEST_DATA,100);
            try {
                cpPredictor = new CompilerPredictor(FILENAME_MODEL);
            } catch (Exception e){
                e.printStackTrace();
            }
            onnxPredictor = new OnnxPredictor(FILENAME_MODEL);
            tribuoPredictor = new TribuoPredictor(FILENAME_MODEL, INPUT_NAME, INPUT_SIZE, 5);
        }

        private float[][] readTestDataFromCSV(String filename, int numRows) {
            float[][] floats = null;
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


    @Benchmark
    public float[][] predictWithCP(PredictorProvider pp){
        var features = pp.features;
        float[][] output = new float[features.length][features[0].length];
        var predictor = pp.cpPredictor;
        for (int i = 0; i < features.length;i++){
            output[i] = predictor.predict(features[i]);
        }
        return output;
    }


    @Benchmark
    public float[][] predictWithONNXRuntime(PredictorProvider pp){
        var features = pp.features;
        float[][] output = new float[features.length][features[0].length];
        var predictor = pp.onnxPredictor;
        for (int i = 0; i < features.length;i++){
            output[i] = predictor.predict(features[i]);
        }
        return output;
    }


    @Benchmark
    public float[][] predictWithTribuo(PredictorProvider pp){
        var features = pp.features;
        float[][] output = new float[features.length][features[0].length];
        var predictor = pp.tribuoPredictor;
        for (int i = 0; i < features.length;i++){
            output[i] = predictor.predict(features[i]);
        }
        return output;
    }


}
