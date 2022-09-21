package parser;

import activation.Layer;

import java.io.IOException;
import java.util.List;

/**
 * Allows different implementations for parsing infernece model
 * files stored in different formats
 */
public abstract class ModelParser {

    /**
     * Takes the path of a file storing an inference model as String, parses this file
     * and returns a list of layers which represent the neural network
     *
     * @param modelFilePath path to the file in which an inference model is stored
     * @return a list of parsed layers which represents the neural network
     */
    public abstract List<Layer> parseModel(String modelFilePath) throws IOException, LayerNotImplementedException, ModelFormatException;
}
