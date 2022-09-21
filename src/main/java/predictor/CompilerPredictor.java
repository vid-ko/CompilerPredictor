package predictor;

import activation.*;
import parser.LayerNotImplementedException;
import parser.ModelFormatException;
import parser.OnnxModelParser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Allows making predictions on a set of features given as float array
 * with an inference model stored in ONNX format
 */
public class CompilerPredictor extends Predictor {

    // all layers of the neural network are stored in here
    private List<Layer> layers;

    // outputs stored during prediction
    private final Map<String,float[]> outputMap;


    /**
     * Constructor which initializes the Neural Network based on the
     * specified ONNX model
     *
     * @param modelFilePath specifies the path to the .onnx file
     */
    public CompilerPredictor(String modelFilePath) throws IOException, LayerNotImplementedException, ModelFormatException {
        layers = new ArrayList<>();
        outputMap = new HashMap<>();
        initInferenceModel(modelFilePath);
    }


    @Override
    public float[] predict(float[] features){
        var nextInput = features;

        for (Layer layer : layers){
            if(layer instanceof Add addLayer){
                // take output from map with the name stored in the layer as input
                addLayer.setInput1(outputMap.get(addLayer.getInput1Name()));
            }
            nextInput = layer.apply(nextInput); // output as input for the next layer
            outputMap.put(layer.outputName,nextInput);
        }
        return nextInput;
    }


    /**
     * Initializes the list of layer by calling the
     * OnnxModelParser's parse model method
     *
     * @param modelFilePath path to .onnx file
     */
    private void initInferenceModel(String modelFilePath) throws IOException, LayerNotImplementedException, ModelFormatException {
        layers = new OnnxModelParser().parseModel(modelFilePath);
    }

}
