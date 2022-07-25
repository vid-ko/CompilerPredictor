package activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class BatchNorm extends Layer {

    private final INDArray weightMatrix;
    private final INDArray biasMatrix;
    private final INDArray meanMatrix;
    private final INDArray varMatrix;
    private static final float EPSILON = 0.000009999999747378752f;

    public BatchNorm(float[] weights, float[] bias, float[] mean, float[] var) {
        weightMatrix = Nd4j.create(weights);
        biasMatrix = Nd4j.create(bias);
        meanMatrix = Nd4j.create(mean);
        varMatrix = Nd4j.create(var);
    }

    @Override
    public float[] apply(float[] input) {
        INDArray inputMatrix = Nd4j.create(input);

        INDArray result = Nd4j.diag(weightMatrix).mmul(inputMatrix.sub(meanMatrix).div(Transforms.sqrt(varMatrix.add(EPSILON)))).add(biasMatrix);

        return result.toFloatVector();
    }
}
