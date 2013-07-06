package tv.floe.metronome.classification.neuralnetworks.transfer;

public class Sigmoid extends TransferFunction {

	private double slope = 1d;
		
	public Sigmoid() {
	}

	public Sigmoid(double slope) {
		this.slope = slope;
	}

	
	public double getSlope() {
		return this.slope;
	}

	public void setSlope(double slope) {
		this.slope = slope;
	}

	/**
	 * Output
	 * 
	 * output == 1/(1+ e^(-slope*input))
	 * 
	 */
	@Override
	public double getOutput(double net) {

		
        if (net > 100) {

        	return 1.0;
        
        } else if (net < -100) {

        	return 0.0;

        }

		double den = 1d + Math.exp(-this.slope * net);
                
		this.output = (1d / den); 
                
		return this.output;
	}

	@Override
	public double getDerivative(double net) { // remove net parameter? maybe we dont need it since we use cached output value
                // +0.1 is fix for flat spot see http://www.heatonresearch.com/wiki/Flat_Spot
		double derivative = this.slope * this.output * (1d - this.output) + 0.1;
		
		return derivative;
	}	
	
	

}
