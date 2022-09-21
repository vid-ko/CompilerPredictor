package activation;

/**
 * Implements the ReLu layer to replace
 * negative values by zero in the apply method
 */
public class Relu extends Layer {


    @Override
    public float[] apply(float[] input) {
        // replaces all negative values by 0
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++){
            output[i] = Math.max(0,input[i]);
        }
        return output;
    }
}
