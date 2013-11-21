package tv.floe.metronome.classification.neuralnetworks.learning.adagrad;

public class AdagradLearningRate {

	private double gamma = 10; // default for gamma (this is the numerator)
	private double squaredGradientSum = 0;
	
	public AdagradLearningRate(double gamma) {
		this.gamma = gamma;
	}
	
	/**
	 * square gradient, then add to ongoing sum
	 * 
	 * @param gradient
	 */
	public void addLastIterationGradient(double gradient) {
		
		this.squaredGradientSum += gradient * gradient;
		
		
	}
	
	/**
	 * eta == gamma / Math.sqrt( this.squaredGradientSum )
	 * 
	 * @return
	 */
	public double compute() {
		if ( this.squaredGradientSum > 0) {
			return this.gamma / Math.sqrt( this.squaredGradientSum );
		} else {
			return this.gamma;
		}
	}

}
