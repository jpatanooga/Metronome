package tv.floe.metronome.classification.neuralnetworks.transfer;

public abstract class TransferFunction {

    protected double output; 

	abstract public double getOutput(double net);

	public double getDerivative(double net) {
		return 1d;
	}
	
	
}
