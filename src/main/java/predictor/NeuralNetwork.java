package predictor;

import activation.*;
import com.google.gson.JsonArray;
import com.google.gson.JsonParser;
import onnx.Onnx;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    //private Layer[] layers;
    private final List<Layer> layers = new ArrayList<>();


    public NeuralNetwork(Layer[] layers){
        //this.layers = layers;
    }

    /**
     * Initializes the Neural Network based on the
     * specified ONNX model
     *
     * @param pathToOnnxFile specifies the path to the .onnx file
     */
    public NeuralNetwork(String pathToOnnxFile){
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
        try{
            var model = Onnx.ModelProto.parseFrom(new FileInputStream(pathToOnnxFile));

            for(var node : model.getGraph().getNodeList()){
                Onnx.TensorProto tensor;
                switch (node.getOpType()){
                    case "Sub":
                        tensor = model.getGraph().getInitializerList().stream()
                                .filter(i -> i.getName().equals(node.getInputList().get(1)))
                                .findAny()
                                .orElse(null);
                        layers.add(new Sub(get1DParamsFromTensor(tensor)));
                        break;
                    case "Div":
                        tensor = model.getGraph().getInitializerList().stream()
                                .filter(i -> i.getName().equals(node.getInputList().get(1)))
                                .findAny()
                                .orElse(null);
                        layers.add(new Div(get1DParamsFromTensor(tensor)));
                        break;
                    case "Gemm":
                        tensor = model.getGraph().getInitializerList().stream()
                                .filter(i -> i.getName().equals(node.getInputList().get(1)))
                                .findAny()
                                .orElse(null);
                        float[][] weights = get2DParamsFromTensor(tensor);
                        tensor = model.getGraph().getInitializerList().stream()
                                .filter(i -> i.getName().equals(node.getInputList().get(2)))
                                .findAny()
                                .orElse(null);

                        float[] bias = get1DParamsFromTensor(tensor);
                        layers.add(new Gemm(weights, bias));
                        break;
                    case "Relu":
                        layers.add(new Relu());
                        break;
                    case "BatchNormalization":
                        tensor = model.getGraph().getInitializerList().stream()
                                .filter(i -> i.getName().equals(node.getInputList().get(1)))
                                .findAny()
                                .orElse(null);
                        float[] batchNormWeights = get1DParamsFromTensor(tensor);
                        tensor = model.getGraph().getInitializerList().stream()
                                .filter(i -> i.getName().equals(node.getInputList().get(2)))
                                .findAny()
                                .orElse(null);
                        float[] batchNormBias = get1DParamsFromTensor(tensor);
                        tensor = model.getGraph().getInitializerList().stream()
                                .filter(i -> i.getName().equals(node.getInputList().get(3)))
                                .findAny()
                                .orElse(null);
                        float[] mean = get1DParamsFromTensor(tensor);
                        tensor = model.getGraph().getInitializerList().stream()
                                .filter(i -> i.getName().equals(node.getInputList().get(4)))
                                .findAny()
                                .orElse(null);
                        float[] var = get1DParamsFromTensor(tensor);
                        layers.add(new BatchNorm(batchNormWeights, batchNormBias, mean, var));
                        break;
                    default: // TODO: raise LayerNotFoundException

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
/*
    private void init(){
        this.layers = new Layer[]{
                new Sub(read1DParamsFromFile("mean")),
                new Div(read1DParamsFromFile("54")),
                new Gemm(read2DParamsFromFile("features.features.0.weight"),read1DParamsFromFile("features.features.0.bias")),
                new Relu(),
                new Gemm(read2DParamsFromFile("features.features.2.weight"),read1DParamsFromFile("features.features.2.bias")),
                new BatchNorm(read1DParamsFromFile("features.features.3.weight"),read1DParamsFromFile("features.features.3.bias"),
                        read1DParamsFromFile("features.features.3.running_mean"),read1DParamsFromFile("features.features.3.running_var")),
                new Relu(),
                new Gemm(read2DParamsFromFile("features.features.5.weight"),read1DParamsFromFile("features.features.5.bias")),
                new Relu(),
                new Gemm(read2DParamsFromFile("features.features.7.weight"),read1DParamsFromFile("features.features.7.bias")),
                new BatchNorm(read1DParamsFromFile("features.features.8.weight"),read1DParamsFromFile("features.features.8.bias"),
                        read1DParamsFromFile("features.features.8.running_mean"),read1DParamsFromFile("features.features.8.running_var")),
                new Relu(),
                new Gemm(read2DParamsFromFile("features.features.10.weight"),read1DParamsFromFile("features.features.10.bias")),
                new Relu(),
                new Gemm(read2DParamsFromFile("features.features.12.weight"),read1DParamsFromFile("features.features.12.bias")),
                new Relu(),
                new Gemm(read2DParamsFromFile("features.features.14.weight"),read1DParamsFromFile("features.features.14.bias")),
                new Relu(),
                new Gemm(read2DParamsFromFile("features.features.16.weight"),read1DParamsFromFile("features.features.16.bias")),
                new Relu(),
                new Gemm(read2DParamsFromFile("features.features.18.weight"),read1DParamsFromFile("features.features.18.bias"))
        };
    }

    private float[][] read2DParamsFromFile(String filename){
        FileReader reader;
        float[][] floats = null;
        try {
            reader = new FileReader("src/main/java/predictor/"+filename+".json");
            JsonArray jsonArray = (JsonArray) new JsonParser().parse(reader);
            floats = new float[jsonArray.size()][jsonArray.get(0).getAsJsonArray().size()];
            for(int i = 0; i < jsonArray.size(); i++){
                for (int j = 0; j < jsonArray.get(0).getAsJsonArray().size(); j++){
                    floats[i][j] = (jsonArray.get(i).getAsJsonArray().get(j).getAsFloat());
                }
            }
        } catch (IOException e){
            e.printStackTrace();
        }
        return floats;
    }


    private float[] read1DParamsFromFile(String filename){
        FileReader reader;
        float[] floats = null;
        try {
            reader = new FileReader("src/main/java/predictor/"+filename+".json");
            JsonArray jsonArray = (JsonArray) new JsonParser().parse(reader);
            floats = new float[jsonArray.size()];
            for(int i = 0; i < jsonArray.size(); i++){
                floats[i] = (jsonArray.get(i).getAsFloat());
            }
        } catch (IOException e){
            e.printStackTrace();
        }
        return floats;
    }

    */

}
