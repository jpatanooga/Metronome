package tv.floe.metronome.classification.neuralnetworks.activation;

import java.io.Serializable;

/**
 * 
 * 
 * @author josh
 *
 */
public abstract class ActivationFunction implements Serializable {

    protected double output; 

	abstract public double getOutput(double net);

	public double getDerivative(double net) {
		return 1d;
	}
	
	
}
