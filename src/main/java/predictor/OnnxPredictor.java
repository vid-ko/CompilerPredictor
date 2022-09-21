package predictor;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.HashMap;
import java.util.Map;

/**
 * Wrapper class to make predictions on a set of features given as float array
 * by using the ONNX Runtime
 */
public class OnnxPredictor extends Predictor implements AutoCloseable{


    private OrtSession session;
    private OrtEnvironment env;


    public OnnxPredictor(String modelFilePath){
        initInferenceModel(modelFilePath);
    }


    @Override
    public float[] predict(float[] features) {
        float[] output = null;
        try {
            float[][] input = new float[1][features.length];
            for (int i = 0; i < features.length; i++){
                input[0][i] = features[i];
            }
            OnnxTensor tensor = OnnxTensor.createTensor(env, input);
            Map<String, OnnxTensor> inputMap = new HashMap<>();
            inputMap.put(session.getInputNames().toArray()[0].toString(), tensor);

            try (var result = session.run(inputMap)) {
                float[][] res = (float[][])result.get(0).getValue();
                output = res[0];
            }

        } catch (OrtException e) {
            e.printStackTrace();
        }
        return output;
    }


    private void initInferenceModel(String modelFilePath) {
        this.env = OrtEnvironment.getEnvironment();
        try {
            this.session = env.createSession(modelFilePath, new OrtSession.SessionOptions());
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }


    @Override
    public void close() throws Exception {
        session.close();
    }
}
