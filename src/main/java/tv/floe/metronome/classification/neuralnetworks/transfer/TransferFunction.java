package tv.floe.metronome.classification.neuralnetworks.transfer;

import java.io.Serializable;

/**
 * AKA "activation function"
 * 
 * @author josh
 *
 */
public abstract class TransferFunction implements Serializable {

    protected double output; 

	abstract public double getOutput(double net);

	public double getDerivative(double net) {
		return 1d;
	}
	
	
}
