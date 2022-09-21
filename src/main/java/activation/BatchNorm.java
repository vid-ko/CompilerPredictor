package activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Implements the BatchNorm layer to perform batch normalization
 * on the incoming data
 */
public class BatchNorm extends Layer {

    private final INDArray biasMatrix;
    private final INDArray meanMatrix;
    private final INDArray varMatrix;
    private final INDArray diagMatrix;
    private INDArray rtMatrix; // to store sqrt(var + epsilon)

    private float epsilon = 0;

    /**
     * Initializes the INDArrays based on the
     * passed parameters
     *
     * @param weights weights as float[]
     * @param bias bias values as float[]
     * @param mean running mean as float[]
     * @param var running variance as float[]
     */
    public BatchNorm(float[] weights, float[] bias, float[] mean, float[] var) {
        var weightMatrix = Nd4j.create(weights);
        biasMatrix = Nd4j.create(bias);
        meanMatrix = Nd4j.create(mean);
        varMatrix = Nd4j.create(var);
        diagMatrix = Nd4j.diag(weightMatrix);
        //prepare here to improve prediction runtime
        rtMatrix = Transforms.sqrt(varMatrix.add(epsilon));
    }

    @Override
    public float[] apply(float[] input) {
        // uses Nd4j to perform batch normalization
        INDArray inputMatrix = Nd4j.create(input);
        INDArray result = diagMatrix
                .mmul(inputMatrix.sub(meanMatrix)
                .div(rtMatrix)).add(biasMatrix);

        return result.toFloatVector();
    }

    /**
     * Sets the epsilon
     *
     * @param epsilon epsilon as float
     */
    public void setEpsilon(float epsilon) {
        this.epsilon = epsilon;
        rtMatrix = Transforms.sqrt(varMatrix.add(epsilon));
    }
}
