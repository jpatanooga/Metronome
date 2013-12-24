package tv.floe.metronome.deeplearning.rbm;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;



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
	 * 
	 * Current Major Questions
	 * - why keep the probabilites around?
	 * - why call the matrix "means" instead of "probabilities" ?
	 * 
	 * @param k
	 */
	public void contrastiveDivergence(int k, Matrix input) {

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
		Pair<Matrix, Matrix> hiddenProbsAndSamplesStart = this.sampleHiddenGivenVisible( input );
		
		Matrix hiddenSample = null;
		
		Pair<Pair<Matrix, Matrix>,Pair<Matrix, Matrix>> gibbsSamplingMatrices = null;
		
		// now run k full steps of alternating Gibbs sampling
		
		for ( int x = 0; x < k; x++ ) {
			
			if (0 == x) {
				
				gibbsSamplingMatrices = this.gibbsSamplingStepFromHidden( hiddenProbsAndSamplesStart.getSecond() );
								
			} else {
				
				gibbsSamplingMatrices = this.gibbsSamplingStepFromHidden( hiddenSample );
				
			}
			
		}
		
		// end of gibbs sampling (k full alternating sampling passes)
		
		// ----- now calculate equation (9) to get the weight changes ------

		// TODO: document this section WRT Hinton math/equations
		
		// now compute the <vi hj>data		
		Matrix trainingDataTimesInitialHiddenStates = input.transpose().times( hiddenProbsAndSamplesStart.getSecond() );

		// now compute the <vi hj>model
//		DoubleMatrix nvSamplesTTimesNhMeans = nvSamples.transpose().mmul(nhMeans);
		// .getSecond().getFirst()
		Matrix nvSamplesTTimesNhMeans = gibbsSamplingMatrices.getFirst().getSecond().transpose().times( gibbsSamplingMatrices.getSecond().getFirst() );
		
		
		// calc the delta between: data - model
//		DoubleMatrix diff = inputTimesPhSample.sub(nvSamplesTTimesNhMeans);
		Matrix dataModelDelta = trainingDataTimesInitialHiddenStates.minus(nvSamplesTTimesNhMeans);
		
		// learningRate * delta(data - model)
		// we're simply updating the connection weights at this point
		Matrix connectionWeightChanges = dataModelDelta.times(this.learningRate);
		
		// ---- end of equation (9) section -----------------
		
		// update the connection weights
		this.connectionWeights = this.connectionWeights.plus(connectionWeightChanges);

		//
		// TODO: review this area - the bias for both hidden and visible 
		// are update differently, review this.
		//

//		DoubleMatrix  vBiasAdd = MatrixUtil.mean(this.input.sub(nvSamples), 0).mul(learningRate);
//		vBias = vBiasAdd.mul(learningRate);
		Matrix vBiasAdd = MatrixUtils.mean( input.minus( gibbsSamplingMatrices.getFirst().getSecond() ) , 0).times(this.learningRate);
		this.visibleBiasNeurons = vBiasAdd.times(learningRate);

//		DoubleMatrix hBiasAdd = MatrixUtil.mean(probHidden.getSecond().sub(nhMeans), 0).mul(learningRate);


		Matrix hBiasAdd = MatrixUtils.mean( hiddenProbsAndSamplesStart.getSecond().minus( gibbsSamplingMatrices.getSecond().getFirst() ) , 0); //.times(this.learningRate);
		this.hiddenBiasNeurons = this.hiddenBiasNeurons.plus( hBiasAdd.times(this.learningRate) );
		
		
		
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
	 * the hidden units, 
	 * 
	 * aka "propagate up"
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
	public Pair<Matrix, Matrix> sampleHiddenGivenVisible(Matrix visible) {
				
		Matrix hiddenProbs = this.generateProbabilitiesForHiddenStatesBasedOnVisibleStates(visible);

		Matrix hiddenBinomialSamples = MatrixUtils.genBinomialDistribution(hiddenProbs, 1, this.randNumGenerator);
		
		return new Pair<Matrix, Matrix>(hiddenProbs, hiddenBinomialSamples);
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
		
		Matrix preSigmoid = hidden.times( this.connectionWeights.transpose() );
		preSigmoid = MatrixUtils.addRowVector(preSigmoid, this.visibleBiasNeurons.viewRow(0));

		return MatrixUtils.sigmoid(preSigmoid);
	}
	
	/**
	 * This function infers state of visible units given hidden units
	 * 
	 * @param hidden
	 */
	public Pair<Matrix, Matrix> sampleVisibleGivenHidden(Matrix hidden) {
		

		// dont understand why this is referred to as "mean" in some codebases
		Matrix visibleProb = this.generateProbabilitiesForVisibleStatesBasedOnHiddenStates(hidden);

		Matrix visibleBinomialSample = MatrixUtils.genBinomialDistribution(visibleProb, 1, this.randNumGenerator);

		return new Pair<Matrix, Matrix>(visibleProb, visibleBinomialSample);
		
	}
	
	/**
	 * based on: 
	 * "gibbs_vhv" which performs a step of Gibbs sampling starting from the visible units.
	 * 
	 * TODO: how do we return things?
	 * 
	 */
	public Pair<Pair<Matrix, Matrix>, Pair<Matrix, Matrix>> gibbsSamplingStepFromVisible(Matrix visible) {
	
		Pair<Matrix, Matrix> hiddenProbsAndSamples = this.sampleHiddenGivenVisible(visible);
		Pair<Matrix, Matrix> visibleProbsAndSamples = this.sampleVisibleGivenHidden( hiddenProbsAndSamples.getSecond() );

		return new Pair<Pair<Matrix, Matrix>, Pair<Matrix, Matrix>>(hiddenProbsAndSamples, visibleProbsAndSamples);
	}
	
	/**
	 * based on: "gibbs_hvh"
	 * - This function implements one step of Gibbs sampling, 
	 * starting from the visible state
	 * 
	 * @param hidden
	 */
	public Pair<Pair<Matrix, Matrix>, Pair<Matrix, Matrix>> gibbsSamplingStepFromHidden(Matrix hidden) {
		
		
		Pair<Matrix, Matrix> visibleProbsAndSamples = this.sampleVisibleGivenHidden(hidden);
		Pair<Matrix, Matrix> hiddenProbsAndSamples = this.sampleHiddenGivenVisible(visibleProbsAndSamples.getSecond());
		
		return new Pair<Pair<Matrix, Matrix>, Pair<Matrix, Matrix>>(visibleProbsAndSamples, hiddenProbsAndSamples);
	}
	
	public void computeFreeEnergy(Matrix visibleSample) {
		
		
	}
	
	
	
}
