package predictor;

import activation.*;
import onnx.Onnx;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CompilerPredictor {

    private final List<Layer> layers;
    private final Map<String,float[]> outputMap;


    /**
     * Initializes the Neural Network based on the
     * specified ONNX model
     *
     * @param pathToOnnxFile specifies the path to the .onnx file
     */
    public CompilerPredictor(String pathToOnnxFile){
        layers = new ArrayList<>();
        outputMap = new HashMap<>();
        initWithOnnxModel(pathToOnnxFile);
    }


    /**
     *
     *
     * @param inputData
     * @return
     */
    public float[] predict(float[] inputData){
        var nextInput = inputData;
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
     * @param pathToOnnxFile
     */
    private void initWithOnnxModel(String pathToOnnxFile){
        // TODO refactoring, make more dynamic, LayerNotFoundException
        try{
            var model = Onnx.ModelProto.parseFrom(new FileInputStream(pathToOnnxFile));
            List<Onnx.TensorProto> initializers = model.getGraph().getInitializerList();

            for(var node : model.getGraph().getNodeList()){
                Onnx.TensorProto tensor;
                switch (node.getOpType()) {
                    case  "Add" -> {
                        // TODO: implement
                        Layer addLayer = new Add(node.getInput(0));
                        addLayer.outputName = node.getOutput(0);
                        layers.add(addLayer);
                    }
                    case "Sub" -> {
                        tensor = initializers.stream()
                                .filter(i -> i.getName().equals(node.getInput(1)))
                                .findAny()
                                .orElse(null);
                        if(tensor != null){
                            Layer subLayer = new Sub(get1DParamsFromTensor(tensor));
                            subLayer.outputName = node.getOutput(0);
                            layers.add(subLayer);
                        }
                    }
                    case "Div" -> {
                        tensor = initializers.stream()
                                .filter(i -> i.getName().equals(node.getInput(1)))
                                .findAny()
                                .orElse(null);
                        if(tensor != null){
                            Layer divLayer = new Div(get1DParamsFromTensor(tensor));
                            divLayer.outputName = node.getOutput(0);
                            layers.add(divLayer);
                        }
                    }
                    case "Gemm" -> {
                        tensor = initializers.stream()
                                .filter(i -> i.getName().equals(node.getInput(1)))
                                .findAny()
                                .orElse(null);
                        if(tensor == null) return;
                        float[][] weights = get2DParamsFromTensor(tensor);
                        tensor = initializers.stream()
                                .filter(i -> i.getName().equals(node.getInput(2)))
                                .findAny()
                                .orElse(null);
                        if(tensor == null) return;
                        float[] bias = get1DParamsFromTensor(tensor);
                        Layer gemmLayer = new Gemm(weights, bias);
                        gemmLayer.outputName = node.getOutput(0);
                        layers.add(gemmLayer);
                    }
                    case "Relu" -> {
                        Layer reluLayer = new Relu();
                        reluLayer.outputName = node.getOutput(0);
                        layers.add(reluLayer);
                    }
                    case "BatchNormalization" -> {
                        tensor = initializers.stream()
                                .filter(i -> i.getName().equals(node.getInput(1)))
                                .findAny()
                                .orElse(null);
                        if(tensor == null) return;
                        float[] batchNormWeights = get1DParamsFromTensor(tensor);
                        tensor = model.getGraph().getInitializerList().stream()
                                .filter(i -> i.getName().equals(node.getInput(2)))
                                .findAny()
                                .orElse(null);
                        if(tensor == null) return;
                        float[] batchNormBias = get1DParamsFromTensor(tensor);
                        tensor = initializers.stream()
                                .filter(i -> i.getName().equals(node.getInput(3)))
                                .findAny()
                                .orElse(null);
                        if(tensor == null) return;
                        float[] mean = get1DParamsFromTensor(tensor);
                        tensor = initializers.stream()
                                .filter(i -> i.getName().equals(node.getInput(4)))
                                .findAny()
                                .orElse(null);
                        if(tensor == null) return;
                        float[] var = get1DParamsFromTensor(tensor);
                        var batchNorm = new BatchNorm(batchNormWeights, batchNormBias, mean, var);
                        batchNorm.setEpsilon(node.getAttribute(0).getF());
                        batchNorm.outputName = node.getOutput(0);
                        layers.add(batchNorm);
                    }
                    default -> {
                    }

                }
            }
        }  catch (IOException e) {
            e.printStackTrace();
        }

    }



    /**
     *
     * @param tensor
     * @return
     */
    private float[] get1DParamsFromTensor(Onnx.TensorProto tensor){
        var rawData = tensor.getRawData();
        ByteBuffer buf = ByteBuffer.wrap(rawData.toByteArray());
        buf.order(ByteOrder.LITTLE_ENDIAN);
        var floatBuf = buf.asFloatBuffer();
        float[] floats = new float[rawData.size()/4];
        for(int i = 0; i < rawData.size()/4; i++){
            floats[i] = floatBuf.get(i);
        }
        return floats;
    }

    private float[][] get2DParamsFromTensor(Onnx.TensorProto tensor){
        var rawData = tensor.getRawData();
        ByteBuffer buf = ByteBuffer.wrap(rawData.toByteArray());
        buf.order(ByteOrder.LITTLE_ENDIAN);
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
