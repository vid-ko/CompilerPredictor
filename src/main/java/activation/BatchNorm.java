package activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class BatchNorm extends Layer {

    private final INDArray biasMatrix;
    private final INDArray meanMatrix;
    private final INDArray varMatrix;

    private final INDArray diagMatrix;
    private INDArray rtMatrix;

    private float epsilon = 0;

    public BatchNorm(float[] weights, float[] bias, float[] mean, float[] var) {
        var weightMatrix = Nd4j.create(weights);
        biasMatrix = Nd4j.create(bias);
        meanMatrix = Nd4j.create(mean);
        varMatrix = Nd4j.create(var);

        diagMatrix = Nd4j.diag(weightMatrix);
        rtMatrix = Transforms.sqrt(varMatrix.add(epsilon));
    }

    @Override
    public float[] apply(float[] input) {
        INDArray inputMatrix = Nd4j.create(input);

        INDArray result = diagMatrix
                .mmul(inputMatrix.sub(meanMatrix)
                .div(rtMatrix)).add(biasMatrix);

        return result.toFloatVector();
    }

    public void setEpsilon(float epsilon) {
        this.epsilon = epsilon;
        rtMatrix = Transforms.sqrt(varMatrix.add(epsilon));
    }
}
