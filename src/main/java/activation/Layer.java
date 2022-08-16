package activation;

public abstract class Layer {

    public String outputName;

    /**
     *  Applies layer specific matrix operations to the input
     *  and returns the result
     *
     * @param input incoming test data or output of a previous layer
     * @return output of this layer
     */
    public abstract float[] apply(float[] input);

}
