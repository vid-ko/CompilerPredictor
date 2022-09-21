package activation;

/**
 * Implements the Div layer to perform element-wise
 * division with the apply method
 */
public class Div extends Layer {

    private final float[] weights;

    public Div(float[] weights){
        this.weights = weights;
    }

    @Override
    public float[] apply(float[] input) {
        // element-wise division of the input by the weights
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++){
            output[i] = input[i] / weights[i];
        }
        return output;
    }
}
