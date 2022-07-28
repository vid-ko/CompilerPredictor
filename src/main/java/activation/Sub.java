package activation;

public class Sub extends Layer {

    private final float[] weights;

    public Sub(float[] weights){
        this.weights = weights;
    }

    @Override
    public float[] apply(float[] input) {

        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++){
            output[i] = input[i] - weights[i];
        }
        return output;
    }
}
