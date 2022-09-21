package predictor;

/**
 * Allows making predictions on a set of features given as float array
 */
public abstract class Predictor {

    /**
     * Takes input features in form of a float array and returns the
     * prediction result as float array.
     *
     * @param features input features in a float array
     * @return prediction result as float array
     */
    public abstract float[] predict(float[] features);
}
