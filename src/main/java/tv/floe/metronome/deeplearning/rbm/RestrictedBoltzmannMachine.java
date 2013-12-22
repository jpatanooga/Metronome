package tv.floe.metronome.deeplearning.rbm;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;



/**
 * Based on work by Hinton, et al 2006
 * 
 * And inspired by the RBM Implementation of Adam Gibson:
 * 
 * https://github.com/agibsonccc/java-deeplearning
 * 
 * 
 * @author josh
 *
 */
public class RestrictedBoltzmannMachine {
	
	private double learningRate = 0.1d;
	
	public int numberVisibleNeurons;
	public int numberHiddenNeurons;
	
	public Matrix connectionWeights;
	
	public Matrix hiddenBiasNeurons;
	public Matrix visibleBiasNeurons;
	
	public RandomGenerator randNumGenerator;
	
	public Matrix trainingDataset;

	
	public void contrastiveDivergence(int k) {

	}
	
	public void setLearningRate(double alpha) {
		
		this.learningRate = alpha;
		
	}
	
	public double getReConstructionCrossEntropy() {

		return 0;
	}
	
	public void sampleHiddenGivenVisible(Matrix visible) {
		
	}

	public void sampleVisibleGivenHidden(Matrix hidden) {
		
	}
	
	/*
	 * 
	public DoubleMatrix propUp(DoubleMatrix v) {
		DoubleMatrix preSig = v.mmul(W);
		preSig = preSig.addRowVector(hBias);
		return MatrixUtil.sigmoid(preSig);

	}

	 * 
	 */

	/**
	 * 
	 * 
	 * @param visible
	 * @return
	 */
	public Matrix propUp(Matrix visible) {
		
		Matrix preSigmoid = visible.times( this.connectionWeights );
		
	}
	
}
