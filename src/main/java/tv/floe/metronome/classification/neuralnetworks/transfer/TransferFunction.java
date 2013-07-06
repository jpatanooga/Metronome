package tv.floe.metronome.classification.neuralnetworks.transfer;

public abstract class TransferFunction {

    protected double output; // cached output value to avoid double calculation for derivative

	/**
	 * Returns the ouput of this function.
	 * 
	 * @param net
	 *            net input
	 */
	abstract public double getOutput(double net);

	/**
	 * Returns the first derivative of this function.
	 * 
	 * @param net
	 *            net input
	 */
	public double getDerivative(double net) {
		return 1d;
	}
	
	
}
