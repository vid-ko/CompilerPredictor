package parser;

import activation.*;
import onnx.Onnx;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Allows to parse an .onnx file with the use of the
 * method parseModel(String modelFilePath)
 */
public class OnnxModelParser extends ModelParser {

    /**
     * Standard input position in a node for the name of the weights initializer
     */
    private static final int INPUT_POS_WEIGHTS = 1;

    /**
     * Standard input position in a node for the name of the bias initializer
     */
    private static final int INPUT_POS_BIASES = 2;

    /**
     * Standard input position in a node for the name of the mean initializer
     */
    private static final int INPUT_POS_MEAN = 3;

    /**
     * Standard input position in a node for the name of the var initializer
     */
    private static final int INPUT_POS_VAR = 4;

    /**
     * Standard input position in a node for the name of the
     * initializer in which the output of the previous layer is stored
     */
    private static final int INPUT_POS_LAYER_OUTPUT = 0;

    /**
     * Standard output position in a node for the name of the initializer
     * in which the output is stored
     */
    private static final int STANDARD_OUTPUT_POS = 0;

    /**
     * Standard attribute position in a node where the epsilon is stored
     */
    private static final int ATTRIBUTE_POS_EPSILON = 0;


    @Override
    public List<Layer> parseModel(String modelFilePath) throws IOException, LayerNotImplementedException, ModelFormatException {
        List<Layer> layers = new ArrayList<>();

        var model = Onnx.ModelProto.parseFrom(new FileInputStream(modelFilePath));
        List<Onnx.TensorProto> initializers = model.getGraph().getInitializerList();

        for(var node : model.getGraph().getNodeList()){
            switch (node.getOpType()) {
                case  "Add" -> {
                    Layer addLayer = new Add(node.getInput(INPUT_POS_LAYER_OUTPUT));
                    addLayer.outputName = node.getOutput(STANDARD_OUTPUT_POS);
                    layers.add(addLayer);
                }
                case "Sub" -> {
                    var tensor = getTensorByName(initializers, node.getInput(INPUT_POS_WEIGHTS));
                    if(tensor != null){
                        Layer subLayer = new Sub(get1DParamsFromTensor(tensor));
                        subLayer.outputName = node.getOutput(STANDARD_OUTPUT_POS);
                        layers.add(subLayer);
                    } else {
                        throw new ModelFormatException("Tensor not found: " + node.getInput(INPUT_POS_WEIGHTS));
                    }
                }
                case "Div" -> {
                    var tensor = getTensorByName(initializers, node.getInput(INPUT_POS_WEIGHTS));
                    if(tensor != null){
                        Layer divLayer = new Div(get1DParamsFromTensor(tensor));
                        divLayer.outputName = node.getOutput(STANDARD_OUTPUT_POS);
                        layers.add(divLayer);
                    } else {
                        throw new ModelFormatException("Tensor not found: " + node.getInput(INPUT_POS_WEIGHTS));
                    }
                }
                case "Gemm" -> {
                    var weightsTensor = getTensorByName(initializers, node.getInput(INPUT_POS_WEIGHTS));
                    if(weightsTensor == null) {
                        throw new ModelFormatException("Tensor not found: " + node.getInput(INPUT_POS_WEIGHTS));
                    }
                    float[][] weights = get2DParamsFromTensor(weightsTensor);
                    var biasTensor = getTensorByName(initializers, node.getInput(INPUT_POS_BIASES));
                    if(biasTensor == null) {
                        throw new ModelFormatException("Tensor not found: " + node.getInput(INPUT_POS_BIASES));
                    }
                    float[] bias = get1DParamsFromTensor(biasTensor);
                    Layer gemmLayer = new Gemm(weights, bias);
                    gemmLayer.outputName = node.getOutput(STANDARD_OUTPUT_POS);
                    layers.add(gemmLayer);
                }
                case "Relu" -> {
                    Layer reluLayer = new Relu();
                    reluLayer.outputName = node.getOutput(STANDARD_OUTPUT_POS);
                    layers.add(reluLayer);
                }
                case "BatchNormalization" -> {
                    var weightsTensor = getTensorByName(initializers, node.getInput(INPUT_POS_WEIGHTS));
                    if(weightsTensor == null) {
                        throw new ModelFormatException("Tensor not found: " + node.getInput(INPUT_POS_WEIGHTS));
                    }
                    float[] batchNormWeights = get1DParamsFromTensor(weightsTensor);
                    var biasTensor = getTensorByName(initializers, node.getInput(INPUT_POS_BIASES));
                    if(biasTensor == null) {
                        throw new ModelFormatException("Tensor not found: " + node.getInput(INPUT_POS_BIASES));
                    }
                    float[] batchNormBias = get1DParamsFromTensor(biasTensor);
                    var meanTensor = getTensorByName(initializers, node.getInput(INPUT_POS_MEAN));
                    if(meanTensor == null) {
                        throw new ModelFormatException("Tensor not found: " + node.getInput(INPUT_POS_MEAN));
                    }
                    float[] mean = get1DParamsFromTensor(meanTensor);
                    var varTensor = getTensorByName(initializers, node.getInput(INPUT_POS_VAR));
                    if(varTensor == null) {
                        throw new ModelFormatException("Tensor not found: " + node.getInput(INPUT_POS_VAR));
                    }
                    float[] var = get1DParamsFromTensor(varTensor);
                    var batchNorm = new BatchNorm(batchNormWeights, batchNormBias, mean, var);
                    batchNorm.setEpsilon(node.getAttribute(ATTRIBUTE_POS_EPSILON).getF());
                    batchNorm.outputName = node.getOutput(STANDARD_OUTPUT_POS);
                    layers.add(batchNorm);
                }
                default -> {
                    throw new LayerNotImplementedException(node.getOpType());
                }
            }
        }
        return layers;
    }


    /**
     * Returns the TensorProto with the specified name in a list of TensorProto
     *
     * @param tensorList TensorProto list to be searched in
     * @param name name of the TensorProto which should be retrieved from the list
     * @return the found TensorProto with the specified name or null if the list does
     *         not contain a TensorProto with the specified name
     */
    private Onnx.TensorProto getTensorByName(List<Onnx.TensorProto> tensorList, String name ){
        return tensorList.stream()
                .filter(i -> i.getName().equals(name))
                .findAny()
                .orElse(null);
    }



    /**
     * Transforms the raw data of a TensorProto to float[]
     *
     * @param tensor Onnx.TensorProto containing the raw data to be transformed
     * @return Raw data of tensor as float[]
     */
    private float[] get1DParamsFromTensor(Onnx.TensorProto tensor){
        var rawData = tensor.getRawData();
        ByteBuffer buf = ByteBuffer.wrap(rawData.toByteArray());
        buf.order(ByteOrder.LITTLE_ENDIAN); // data stored in little endian
        var floatBuf = buf.asFloatBuffer();
        float[] floats = new float[rawData.size()/4];
        for(int i = 0; i < rawData.size()/4; i++){
            floats[i] = floatBuf.get(i);
        }
        return floats;
    }

    /**
     * Transforms the raw data of a TensorProto to float[][]
     *
     * @param tensor Onnx.TensorProto containing the raw data to be transformed
     * @return Raw data of tensor as float[][]
     */
    private float[][] get2DParamsFromTensor(Onnx.TensorProto tensor){
        var rawData = tensor.getRawData();
        ByteBuffer buf = ByteBuffer.wrap(rawData.toByteArray());
        buf.order(ByteOrder.LITTLE_ENDIAN); // data stored in little endian
        var floatBuf = buf.asFloatBuffer();
        int dim1 = (int)tensor.getDims(0);
        int dim2 = (int)tensor.getDims(1);
        float[][] floats = new float[dim1][dim2];
        for(int i = 0; i < dim1; i++){
            for(int j = 0; j < dim2; j++){
                floats[i][j] = floatBuf.get(i*dim2 + j);
            }
        }
        return floats;
    }
}
