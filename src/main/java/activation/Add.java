package activation;

public class Add extends Layer{

    private final float[] input1;

    public Add(float[] input1){
        this.input1 = input1;
    }

    @Override
    public float[] apply(float[] input2) {
        float[] output = new float[input2.length];
        for (int i = 0; i < input2.length; i++){
            output[i] = input2[i] + input1[i];
        }
        return output;
    }
}
