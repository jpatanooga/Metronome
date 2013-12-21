package tv.floe.metronome.deeplearning.rbm;



/**
 * Based on work by Hinton, et al 2006
 * 
 * And the RBM Implementation of Adam Gibson:
 * 
 * https://github.com/agibsonccc/java-deeplearning
 * 
 * 
 * @author josh
 *
 */
public class RestrictedBoltzmannMachine {
	
	private double learningRate = 0.1d;

	
	public void contrastiveDivergence(int k) {

	}
	
	public void setLearningRate(double alpha) {
		
		this.learningRate = alpha;
		
	}
	
	public double getReConstructionCrossEntropy() {

		return 0;
	}
	
	public void sampleHiddenGivenVisible() {
		
	}

	public void sampleVisibleGivenHidden() {
		
	}
	
	
}
