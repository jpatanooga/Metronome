package tv.floe.metronome.classification.neuralnetworks.activation;

public class Step  extends ActivationFunction {

	private double dHigh = 1d;	
	private double dLow = 0d;

	public Step() {
	}

	public Step(double high, double low) {

		this.dHigh = high;		
		this.dLow = low;
		
	}

        @Override
	public double getOutput(double net) {
		if (net > 0d)
			return this.dHigh;
		else
			return this.dLow;
	}

	public double getHighVal() {
		return this.dHigh;
	}
	
	public void setHighVal(double high) {
		this.dHigh = high;
	}

	public double getLowVal() {
		return this.dLow;
	}

	public void setLowVal(double low) {
		this.dLow = low;
	}

}
