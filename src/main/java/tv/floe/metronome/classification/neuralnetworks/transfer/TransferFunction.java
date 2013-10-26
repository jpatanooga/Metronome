package tv.floe.metronome.classification.neuralnetworks.transfer;

/**
 * AKA "activation function"
 * 
 * @author josh
 *
 */
public abstract class TransferFunction {

    protected double output; 

	abstract public double getOutput(double net);

	public double getDerivative(double net) {
		return 1d;
	}
	
	
}
