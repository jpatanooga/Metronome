package tv.floe.metronome.deeplearning.rbm;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
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

	/**
	 * CTOR
	 * 
	 * So at some point we make this a more elaborate setup to build RBMs?
	 * 
	 * @param numVisibleNeurons
	 * @param numHiddenNeurons
	 */
	public RestrictedBoltzmannMachine(int numVisibleNeurons, int numHiddenNeurons, RandomGenerator rnd) {
		
		this.numberVisibleNeurons = numVisibleNeurons;
		this.numberHiddenNeurons = numHiddenNeurons;
		
		if (rnd == null) {
			
			this.randNumGenerator = new MersenneTwister(1234);

		} else {
			
			this.randNumGenerator = rnd;
			
		}
		
		double a = 1.0 / (double) this.numberVisibleNeurons;
		
		UniformRealDistribution realDistributionGenerator = new UniformRealDistribution(this.randNumGenerator,-a,a,UniformRealDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);

		this.connectionWeights = new DenseMatrix( this.numberVisibleNeurons, this.numberHiddenNeurons );
		this.connectionWeights.assign(0.0);
		
		for (int r = 0; r < this.connectionWeights.numRows(); r++) {
			
			for(int c = 0; c < this.connectionWeights.numCols(); c++) { 
			
				this.connectionWeights.setQuick( r, c, realDistributionGenerator.sample() );
			
			}

		}
		

 
		this.hiddenBiasNeurons = new DenseMatrix( 1, this.numberHiddenNeurons );
		this.hiddenBiasNeurons.assign(0.0);

		this.visibleBiasNeurons = new DenseMatrix( 1, this.numberVisibleNeurons );
		this.visibleBiasNeurons.assign(0.0);

		
		
		
	}
	
	/**
	 * Based on Hinton's explanation on CD from (Hinton, 2002)
	 * 
	 * 1. start by setting the states of the visible units to a training vector
	 * 
	 * 2. the binary states of the hidden units are all computed in parallel using Equation (7)
	 * 
	 * 3. Once binary states have been chosen for the hidden units, a "reconstruction" is produced by setting each vi to 1
	 * with a probability given by equation (8)
	 * 
	 * 4. The change in weights is given by Equation (9)
	 * 
	 * @param k
	 */
	public void contrastiveDivergence(int k) {

		/*
		 * 
		 * we initialize the Gibbs chain with the hidden sample 
		 * generated during the positive phase, therefore implementing CD. 
		 * Once we have established the starting point of the chain, 
		 * we can then compute the sample at the end of the Gibbs chain, 
		 * sample that we need for getting the gradient
		 */
		
		// init CDk
		
		// do gibbs sampling given V to get the Hidden states based on the training input
		// compute positive phase
		
		Matrix hidden_sample_init = this.sampleHiddenGivenVisible( null );
		
		Matrix hiddenSample = null;
		
		// now run k full steps of alternating Gibbs sampling
		
		for ( int x = 0; x < k; x++ ) {
			
			if (0 == x) {
				
				this.gibbsSamplingStepFromHidden( hidden_sample_init );
				
			} else {
				
				this.gibbsSamplingStepFromHidden( hiddenSample );
				
			}
			
			// TODO: gibbs sampling functions need to 
			// return something and we need to update the matrix samples
			
			
		}
		
		// end of gibbs sampling (k full alternating sampling passes)
		
		// ----- now calculate equation (9) to get the weight changes ------

		// now compute the <vi hj>data		
//		DoubleMatrix inputTimesPhSample =  this.input.transpose().mmul(ph.getSecond());
// TODO: look at how the training dataset x hiddenSample works out wrt matrix sizes
		Matrix trainingDataTimesHiddenStates = this.trainingDataset.transpose().times(hidden_sample_init);

		// now compute the <vi hj>model
//		DoubleMatrix nvSamplesTTimesNhMeans = nvSamples.transpose().mmul(nhMeans);
		
		// data - model
//		DoubleMatrix diff = inputTimesPhSample.sub(nvSamplesTTimesNhMeans);
		
		// learningRate * delta(data - model)
//		DoubleMatrix wAdd = diff.mul(learningRate);
		
		// ---- end of equation (9) section -----------------
		
		
		
	}
	
	public void setLearningRate(double alpha) {
		
		this.learningRate = alpha;
		
	}
	
	public double getReConstructionCrossEntropy() {

		return 0;
	}
	

	
	/**
	 * "Compute Activations"
	 * 
	 * Generate probabilities for each hidden unit being set to 1
	 * Equation (7) in Hinton
	 * 
	 * This function propagates the visible units activation upwards to 
	 * the hidden units, aka "propagate up"
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
	 * This function infers state of hidden units given visible units
	 * 
	 * @param visible
	 */
	public Matrix sampleHiddenGivenVisible(Matrix visible) {
				
		Matrix hiddenProbs = this.generateProbabilitiesForHiddenStatesBasedOnVisibleStates(visible);

		Matrix hiddenBinomialSamples = MatrixUtils.genBinomialDistribution(hiddenProbs, 1, this.randNumGenerator);
		
		return hiddenBinomialSamples;
	}
	
	
	/**
	 * "Compute Activations"
	 * 
	 * Generate probabilities for each visible unit being set to 1 given hidden states
	 * Equation (8) in Hinton
	 * 
	 * This function propagates the hidden units activation downwards to 
	 * the visible units
	 * 
	 * Aka "Propagate Down"
	 * 
	 * TODO: can this also mean "reconstruction" ?
	 * 
	 * @param visible
	 * @return
	 */
	public Matrix generateProbabilitiesForVisibleStatesBasedOnHiddenStates(Matrix hidden) {
		
		Matrix preSigmoid = hidden.times( this.connectionWeights );
		preSigmoid = MatrixUtils.addRowVector(preSigmoid, this.visibleBiasNeurons.viewRow(0));

		return MatrixUtils.sigmoid(preSigmoid);
	}
	
	/**
	 * This function infers state of visible units given hidden units
	 * 
	 * @param hidden
	 */
	public Matrix sampleVisibleGivenHidden(Matrix hidden) {
		

		Matrix visibleProb = this.generateProbabilitiesForVisibleStatesBasedOnHiddenStates(hidden);

		Matrix visibleBinomialSample = MatrixUtils.genBinomialDistribution(visibleProb, 1, this.randNumGenerator);

		return visibleBinomialSample;
		
	}
	
	/**
	 * based on: 
	 * "gibbs_vhv" which performs a step of Gibbs sampling starting from the visible units.
	 * 
	 * TODO: how do we return things?
	 * 
	 */
	public void gibbsSamplingStepFromVisible(Matrix visible) {
	
		Matrix hidden_sampled = this.sampleHiddenGivenVisible(visible);
		Matrix visible_sampled = this.sampleVisibleGivenHidden(hidden_sampled);
		
		
	}
	
	/**
	 * based on: "gibbs_hvh"
	 * - This function implements one step of Gibbs sampling, 
	 * starting from the visible state
	 * 
	 * @param hidden
	 */
	public void gibbsSamplingStepFromHidden(Matrix hidden) {
		
		Matrix visible_sampled = this.sampleVisibleGivenHidden(hidden);
		Matrix hidden_sampled = this.sampleHiddenGivenVisible(visible_sampled);
		
	}
	
	public void computeFreeEnergy(Matrix visibleSample) {
		
		
	}
	
	
	
}
