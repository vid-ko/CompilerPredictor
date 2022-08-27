package predictor;

import activation.*;
import parser.OnnxModelParser;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CompilerPredictor extends Predictor {

    private List<Layer> layers;
    private final Map<String,float[]> outputMap;


    /**
     * Initializes the Neural Network based on the
     * specified ONNX model
     *
     * @param modelFilePath specifies the path to the .onnx file
     */
    public CompilerPredictor(String modelFilePath){
        layers = new ArrayList<>();
        outputMap = new HashMap<>();
        initInferenceModel(modelFilePath);
    }


    @Override
    public float[] predict(float[] features){
        var nextInput = features;
        for (Layer layer : layers){
            if(layer instanceof Add addLayer){
                addLayer.setInput1(outputMap.get(addLayer.getInput1Name()));
            }
            nextInput = layer.apply(nextInput);
            outputMap.put(layer.outputName,nextInput);
        }
        return nextInput;
    }


    /**
     *
     * @param modelFilePath
     */
    private void initInferenceModel(String modelFilePath){
        layers = new OnnxModelParser().parseModel(modelFilePath);
    }

}
