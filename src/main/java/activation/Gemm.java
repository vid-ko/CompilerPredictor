package activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Implements the Linear layer to perform linear
 * transformation on the incoming data by the
 * use of the apply method
 */
public class Gemm extends Layer {

    private final INDArray weightMatrix;
    private final INDArray biasMatrix;

    /**
     * Initializes the INDArrays with
     * the passed weights and bias
     *
     * @param weights weights as float[][]
     * @param bias bias as float[]
     */
    public Gemm(float[][] weights, float[] bias){
        weightMatrix = Nd4j.create(weights);
        biasMatrix = Nd4j.create(bias);

    }

    @Override
    public float[] apply(float[] input) {
        // uses Nd4j to perform linear transformation
        INDArray inputMatrix = Nd4j.create(input);
        INDArray result = weightMatrix.mmul(inputMatrix).add(biasMatrix);
        return result.toFloatVector();
    }
}
