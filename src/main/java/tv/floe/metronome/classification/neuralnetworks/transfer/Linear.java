package tv.floe.metronome.classification.neuralnetworks.transfer;

public class Linear  extends TransferFunction {
	
	private double slope = 1d;

	public Linear() {
	}

	public Linear(double slope) {
		this.slope = slope;
	}


	public double getSlope() {
		return this.slope;
	}

	public void setSlope(double slope) {
		this.slope = slope;
	}

    @Override
	public double getOutput(double net) {
		return slope * net;
	}

	@Override
	public double getDerivative(double net) {
		return this.slope;
	}
}
