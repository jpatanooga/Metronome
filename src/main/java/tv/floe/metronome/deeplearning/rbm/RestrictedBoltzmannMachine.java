package tv.floe.metronome.deeplearning.rbm;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;

import tv.floe.metronome.math.MatrixUtils;



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
	
	/**
	 * Generate probabilities for each hidden unit being set to 1
	 * Equation (7) in Hinton
	 * 
	 * @param visible
	 * @return
	 */
	public Matrix generateProbabilitiesForHiddenStatesBasedOnVisibleStates(Matrix visible) {
		
		Matrix preSigmoid = visible.times( this.connectionWeights );
		preSigmoid = MatrixUtils.addRowVector(preSigmoid, this.hiddenBiasNeurons.viewRow(0));

		return MatrixUtils.sigmoid(preSigmoid);
	}

	/**
	 * Generate probabilities for each visible unit being set to 1 given hidden states
	 * Equation (8) in Hinton
	 * 
	 * @param visible
	 * @return
	 */
	public Matrix generateProbabilitiesForVisibleStatesBasedOnHiddenStates(Matrix hidden) {
		
		Matrix preSigmoid = hidden.times( this.connectionWeights );
		preSigmoid = MatrixUtils.addRowVector(preSigmoid, this.visibleBiasNeurons.viewRow(0));

		return MatrixUtils.sigmoid(preSigmoid);
	}
	
	
}
