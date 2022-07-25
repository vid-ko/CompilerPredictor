package activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Gemm extends Layer {

    private final INDArray weightMatrix;
    private final INDArray biasMatrix;

    public Gemm(float[][] weights, float[] bias){
        weightMatrix = Nd4j.create(weights);
        biasMatrix = Nd4j.create(bias);

    }

    @Override
    public float[] apply(float[] input) {
        INDArray inputMatrix = Nd4j.create(input);

        //INDArray result = inputMatrix.mmul(weightMatrix.transpose()).add(biasMatrix);
        INDArray result = weightMatrix.mmul(inputMatrix).add(biasMatrix);

        return result.toFloatVector();
    }
}
