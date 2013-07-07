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
    public void randomize(NeuralNetwork nn) {
    	
        super.randomize(nn);

        int inNeuronsCnt = nn.getInputNeurons().size();
        int hiddenNeuronsCnt = 0;

        for (int i = 1; i < nn.getLayersCount() - 1; i++) {
        	hiddenNeuronsCnt += nn.getLayerByIndex(i).getNeuronsCount();
        }

        double beta = 0.7 * Math.pow(hiddenNeuronsCnt, 1.0 / inNeuronsCnt); // should we use the total number of hidden neurons or different norm for each layer


        for (Layer layer : nn.getLayers()) {
        	
            double norm = 0.0;
            
            for (Neuron neuron : layer.getNeurons()) {
            
            	for (Connection connection : neuron.getInConnections()) {
                
            		double weight = connection.getWeight().getValue();
                    norm += weight * weight;
                
            	}
            
            }
            
            norm = Math.sqrt(norm);

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
