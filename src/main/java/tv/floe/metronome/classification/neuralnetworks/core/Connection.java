package tv.floe.metronome.classification.neuralnetworks.core;

import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class Connection {

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
//    	System.out.println( ">>> conn > getWeightedOutput : " + this.from.getOutput() + "  * " + weight.value );
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
/*
    public void setFromNeuron(Neuron fromNeuron) throws Exception {
        if (fromNeuron == null) {
            throw new Exception("From neuron in connection cant be null!");
        } else {
            this.from = fromNeuron;
        }
    }
*/
    public Neuron getToNeuron() {
        return to;
    }
/*
    public void setToNeuron(Neuron toNeuron) throws Exception {
        if (toNeuron == null) {
            throw new Exception("From neuron in connection cant be null!");
        } else {
            this.to = toNeuron;
        }
    }    
	*/
    
    public double getInput() {
        return this.from.getOutput();
    }

    
}
