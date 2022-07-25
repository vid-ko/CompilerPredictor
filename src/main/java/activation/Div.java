package activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Div extends Layer {

    private final float[] weights;
    private final INDArray weightMatrix;

    public Div(float[] weights){
        this.weights = weights;
        weightMatrix = Nd4j.create(weights);
    }

    @Override
    public float[] apply(float[] input) {
        // TODO use Nd4j
        //INDArray inputMatrix = Nd4j.create(input);
        //var output = inputMatrix.sub(weightMatrix).toFloatVector();


        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++){
            output[i] = input[i] / weights[i];
        }


        return output;
    }
}
