package activation;

public class Relu extends Layer {


    @Override
    public float[] apply(float[] input) {

        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++){
            output[i] = Math.max(0,input[i]);
        }
        return output;
    }
}
