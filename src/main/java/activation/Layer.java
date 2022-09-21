package activation;

/**
 * Allows the implementation of different neural network layers
 * and performing its operation by calling the apply method
 */
public abstract class Layer {

    /**
     * name to identify the layers output
     */
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
