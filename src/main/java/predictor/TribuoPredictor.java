package predictor;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.tribuo.*;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.impl.ArrayExample;
import org.tribuo.interop.onnx.DenseTransformer;
import org.tribuo.interop.onnx.ONNXExternalModel;
import org.tribuo.interop.onnx.RegressorTransformer;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TribuoPredictor extends Predictor{

    private ONNXExternalModel<Regressor> inferneceModel;
    private RegressionFactory regressionFactory;


    public TribuoPredictor(String modelFilePath, int inputSize, int outputSize){
        initInferenceModel(modelFilePath, inputSize, outputSize);
    }


    @Override
    public float[] predict(float[] features) {
        float[] output = null;

        List<Feature> input = new ArrayList<>();
        for (int i = 0; i < features.length; i++){
            input.add(new Feature("x"+i, features[i]));
        }

        var prov = new SimpleDataSourceProvenance("", regressionFactory);
        Regressor regressor = new Regressor("y",0);
        Example<Regressor> example = new ArrayExample<Regressor>(regressor,input);
        List<Example<Regressor>> list = new ArrayList<Example<Regressor>>();
        list.add(example);
        DataSource<Regressor> datasrc = new ListDataSource<Regressor>(list,regressionFactory, prov);
        var splitter = new TrainTestSplitter<>(datasrc, 0.0f, 0L);
        Dataset<Regressor> evalData = new MutableDataset<>(splitter.getTest());
        var result = inferneceModel.predict(evalData);
        double[] doubles = result.get(0).getOutput().getValues();

        output = new float[doubles.length];
        for(int i = 0; i < doubles.length; i++){
            output[i] = (float)doubles[i];
        }

        return output;
    }


    private void initInferenceModel(String modelFilePath, int inputSize, int outputSize) {
        regressionFactory = new RegressionFactory();

        Map<Regressor, Integer> ptOutMapping = new HashMap<>();
        for (int i = 0; i < outputSize; i++) {
            ptOutMapping.put(new Regressor("y" + i, 0), i);
        }

        var ortEnv = OrtEnvironment.getEnvironment();
        var sessionOpts = new OrtSession.SessionOptions();

        var denseTransformer = new DenseTransformer();

        try {
            Map<String, Integer> ptFeatMapping = new HashMap<>();
            for (int i = 0; i < inputSize; i++) {
                ptFeatMapping.put("x" + i, i);
            }

            inferneceModel = ONNXExternalModel.createOnnxModel(regressionFactory, ptFeatMapping, ptOutMapping,
                    denseTransformer, new RegressorTransformer(), sessionOpts, Paths.get(modelFilePath), "input.1");

        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

}
