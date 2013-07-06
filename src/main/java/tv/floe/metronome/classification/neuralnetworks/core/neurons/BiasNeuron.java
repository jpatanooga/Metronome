package tv.floe.metronome.classification.neuralnetworks.core.neurons;

import tv.floe.metronome.classification.neuralnetworks.core.Connection;
	
public class BiasNeuron extends Neuron {
	
	public BiasNeuron() {
	        super();
	}
	
	@Override
	public double getOutput() {
	    return 1;
	}
	
	@Override
	public void addInConnection(Connection connection) {
	  
	}
	
	@Override
	public void addInConnection(Neuron fromNeuron, double weightVal) { 
	  
	}
	
	@Override
	public void addInConnection(Neuron fromNeuron){ 
	  
	}	
	
}
