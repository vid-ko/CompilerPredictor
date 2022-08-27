package parser;

import activation.Layer;

import java.util.List;

public abstract class ModelParser {

    public abstract List<Layer> parseModel(String modelFilePath);
}
