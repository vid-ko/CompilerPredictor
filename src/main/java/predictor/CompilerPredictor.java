package predictor;

import activation.*;
import onnx.Onnx;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public class CompilerPredictor {

    private final List<Layer> layers;


    /**
     * Initializes the Neural Network based on the
     * specified ONNX model
     *
     * @param pathToOnnxFile specifies the path to the .onnx file
     */
    public CompilerPredictor(String pathToOnnxFile){
        layers = new ArrayList<>();
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
            nextInput = layer.apply(nextInput);
        }
        return nextInput;
    }


    /**
     *
     * @param pathToOnnxFile
     */
    public void initWithOnnxModel(String pathToOnnxFile){
        // TODO refactoring, make more dynamic, LayerNotFoundException
        try{
            var model = Onnx.ModelProto.parseFrom(new FileInputStream(pathToOnnxFile));

            for(var node : model.getGraph().getNodeList()){
                List<Onnx.TensorProto> initializers = model.getGraph().getInitializerList();
                Onnx.TensorProto tensor;
                switch (node.getOpType()) {
                    case  "Add" -> {
                        // TODO: implement
                    }
                    case "Sub" -> {
                        tensor = initializers.stream()
                                .filter(i -> i.getName().equals(node.getInput(1)))
                                .findAny()
                                .orElse(null);
                        if(tensor != null){
                            layers.add(new Sub(get1DParamsFromTensor(tensor)));
                        }
                    }
                    case "Div" -> {
                        tensor = initializers.stream()
                                .filter(i -> i.getName().equals(node.getInput(1)))
                                .findAny()
                                .orElse(null);
                        if(tensor != null){
                            layers.add(new Div(get1DParamsFromTensor(tensor)));
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
                        layers.add(new Gemm(weights, bias));
                    }
                    case "Relu" -> layers.add(new Relu());
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
