package tv.floe.metronome.classification.neuralnetworks.math.random;

import tv.floe.metronome.classification.neuralnetworks.core.Connection;
import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class NguyenWidrowRandomizer extends RangeRandomizer {

    public NguyenWidrowRandomizer(double min, double max) {
        super(min, max);
    }

    @Override
    public void randomize(NeuralNetwork neuralNetwork) {
        super.randomize(neuralNetwork);

        int inputNeuronsCount = neuralNetwork.getInputNeurons().size();
        int hiddenNeuronsCount = 0;

        for (int i = 1; i < neuralNetwork.getLayersCount() - 1; i++) {
            hiddenNeuronsCount += neuralNetwork.getLayerByIndex(i).getNeuronsCount();
        }

        double beta = 0.7 * Math.pow(hiddenNeuronsCount, 1.0 / inputNeuronsCount); // should we use the total number of hidden neurons or different norm for each layer


        for (Layer layer : neuralNetwork.getLayers()) {
            // Calculate the Euclidean Norm for the weights: norm += value * value - suma vadrata tezina u layeru
            double norm = 0.0;
            for (Neuron neuron : layer.getNeurons()) {
                for (Connection connection : neuron.getInConnections()) {
                    double weight = connection.getWeight().getValue();
                    norm += weight * weight;
                }
            }
            norm = Math.sqrt(norm);

            // Rescale the weights using beta and the norm: beta * value / norm            
            for (Neuron neuron : layer.getNeurons()) {
                for (Connection connection : neuron.getInConnections()) {
                    double weight = connection.getWeight().getValue();
                    weight = beta * weight / norm;
                    connection.getWeight().setValue(weight);
                }
            }
        }

    }
	
}
