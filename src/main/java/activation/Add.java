package activation;

/**
 * Implements the Add layer which is used to element-wise
 * add two layer outputs in residual neural networks
 */
public class Add extends Layer{

    private float[] input1;
    private final String input1Name;

    public Add(String input1Name){
        this.input1Name = input1Name;
    }

    public String getInput1Name(){
        return this.input1Name;
    }

    public void setInput1(float[] input1){
        this.input1 = input1;
    }

    @Override
    public float[] apply(float[] input2) {
        // element-wise addition of input1 and input2
        float[] output = new float[input2.length];
        for (int i = 0; i < input2.length; i++){
            output[i] = input2[i] + input1[i];
        }
        return output;
    }
}
