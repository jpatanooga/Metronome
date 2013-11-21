package tv.floe.metronome.classification.neuralnetworks.core;

import java.io.Serializable;

import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class Connection implements Serializable {

	// from Neuron
    protected Neuron from;
    
    // to Neuron
    protected Neuron to;
    
    protected Weight weight;	
    
    public Connection(Neuron fromN, Neuron toN) {
    	
    	this.to = toN;
    	this.from = fromN;
    	this.weight = new Weight(1.0);
    }
    
    public Connection(Neuron fromN, Neuron toN, Weight w) {
    	
    	this.to = toN;
    	this.from = fromN;
    	this.weight = w;
    	
    }

    public Connection(Neuron fromN, Neuron toN, double dw) {
    	
    	this.to = toN;
    	this.from = fromN;
    	this.weight = new Weight(dw);
    	
    }
    
    public double getWeightedInput() {
        return this.from.getOutput() * weight.value;
    }
    
    
    public void setWeight(Weight w) {
    	
    	this.weight = w;
    	
    }
    
    public Weight getWeight() {
    	
    	return this.weight;
    	
    }
    
    public Neuron getFromNeuron() {
        return from;
    }

    public Neuron getToNeuron() {
        return to;
    }
    
    public double getInput() {
        return this.from.getOutput();
    }

    
}
