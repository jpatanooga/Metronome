package tv.floe.metronome.deeplearning.rbm;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;


import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseNeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.optimize.NeuralNetworkOptimizer;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;



/**
 * Basic Implementation of a Restricted Boltzmann Machine
 * 
 * A restricted Boltzmann machine (RBM) is a generative stochastic neural
 *  network that can learn a probability distribution over its set of inputs.
 * 
 * Applications
 * 
 * - dimensionality reduction, classification, collaborative filtering, feature learning and topic modelling
 * 
 * Based on work by Hinton, et al 2006
 * 
 * And inspired by the RBM Implementation of Adam Gibson:
 * 
 * https://github.com/agibsonccc/java-deeplearning
 * 
 * 
 * 
 * @author josh
 *
 */
public class RestrictedBoltzmannMachine extends BaseNeuralNetworkVectorized {
	
	//private double learningRate = 0.1d;
	protected NeuralNetworkOptimizer optimizer;

	public RestrictedBoltzmannMachine(int nVisible, int nHidden, Matrix weights, Matrix hBias, Matrix vBias, RandomGenerator rng) {
		super(nVisible, nHidden, weights, hBias, vBias, rng);
	}
	
	public RestrictedBoltzmannMachine(Matrix input, int nVisible, int nHidden, Matrix weights, Matrix hBias, Matrix vBias, RandomGenerator rng) {
		super(input, nVisible, nHidden, weights, hBias, vBias, rng);
	}
	
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
		
		UniformRealDistribution realDistributionGenerator = new UniformRealDistribution(this.randNumGenerator,-a,a);

		this.connectionWeights = new DenseMatrix( this.numberVisibleNeurons, this.numberHiddenNeurons );
		this.connectionWeights.assign(0.0);
		
		for (int r = 0; r < this.connectionWeights.numRows(); r++) {
			
			for(int c = 0; c < this.connectionWeights.numCols(); c++) { 
			
				this.connectionWeights.setQuick( r, c, realDistributionGenerator.sample() );
			
			}

		}
		

 
		this.hiddenBiasNeurons = new DenseMatrix( 1, this.numberHiddenNeurons );
		// switch to column vector ?
		//this.hiddenBiasNeurons = new DenseMatrix( this.numberHiddenNeurons, 1 );
		this.hiddenBiasNeurons.assign(0.0);

		this.visibleBiasNeurons = new DenseMatrix( 1, this.numberVisibleNeurons );
		// switch to column vector ?
		//this.visibleBiasNeurons = new DenseMatrix( this.numberVisibleNeurons, 1 );
		this.visibleBiasNeurons.assign(0.0);

		System.out.println("Creating RBM: visible: " + this.numberVisibleNeurons + ", hidden: " + this.numberHiddenNeurons );
		
		
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
	public void contrastiveDivergence(double learningRate, int k, Matrix input) {

		/*
		 * 
		 * we initialize the Gibbs chain with the hidden sample 
		 * generated during the positive phase, therefore implementing CD. 
		 * Once we have established the starting point of the chain, 
		 * we can then compute the sample at the end of the Gibbs chain, 
		 * sample that we need for getting the gradient
		 */
		
		this.trainingDataset = input;
				
		// init CDk
		
		// do gibbs sampling given V to get the Hidden states based on the training input
		// compute positive phase
		Pair<Matrix, Matrix> hiddenProbsAndSamplesStart = this.sampleHiddenGivenVisible( input );
				
		Pair<Pair<Matrix, Matrix>,Pair<Matrix, Matrix>> gibbsSamplingMatrices = null;
		
		// now run k full steps of alternating Gibbs sampling
		
		//negative visble "means" or "expected values"
		Matrix negativeVisibleExpectedValues = null;
		//negative value samples
		Matrix negativeVisibleSamples = null;
		//negative hidden means or expected values
		Matrix negativeHiddenExpectedValues = null;
		//negative hidden samples
		Matrix negativeHiddenSamples = null;
		
		for ( int x = 0; x < k; x++ ) {
			
			if (0 == x) {
				
				gibbsSamplingMatrices = this.gibbsSamplingStepFromHidden( hiddenProbsAndSamplesStart.getSecond() );
								
			} else {
				
				gibbsSamplingMatrices = this.gibbsSamplingStepFromHidden( negativeHiddenSamples );
				
			}
			
			// "free energy of the negative phase"
			// now create some easier to use aliases
			negativeVisibleExpectedValues = gibbsSamplingMatrices.getFirst().getFirst();
			negativeVisibleSamples = gibbsSamplingMatrices.getFirst().getSecond();
			negativeHiddenExpectedValues = gibbsSamplingMatrices.getSecond().getFirst();
			negativeHiddenSamples = gibbsSamplingMatrices.getSecond().getSecond();
			
			
		}
				
		// ----- now calculate equation (9) to get the weight changes ------
		
		// now compute the <vi hj>data		
		Matrix trainingDataTimesInitialHiddenStates = input.transpose().times( hiddenProbsAndSamplesStart.getSecond() );

		// now compute the <vi hj>model (this may be vi * phj --- double check)
		Matrix negativeVisibleSamplesTransposeTimesNegHiddenExpValues = negativeVisibleSamples.transpose().times( negativeHiddenExpectedValues );
				
		// calc the delta between: data - model
		Matrix dataModelDelta = trainingDataTimesInitialHiddenStates.minus( negativeVisibleSamplesTransposeTimesNegHiddenExpValues );
		
		// learningRate * delta(data - model)
		Matrix connectionWeightChanges = dataModelDelta.times( learningRate );
		
		// ---- end of equation (9) section -----------------
		
		// update the connection weights and bias terms for visible/hidden units
		this.connectionWeights = this.connectionWeights.plus( connectionWeightChanges );

		Matrix vBiasAdd = MatrixUtils.mean( input.minus( negativeVisibleSamples ) , 0); 
		this.visibleBiasNeurons = this.visibleBiasNeurons.plus( vBiasAdd.times( learningRate ) );

		Matrix hBiasAdd = MatrixUtils.mean( hiddenProbsAndSamplesStart.getSecond().minus( negativeHiddenExpectedValues ) , 0); //.times(this.learningRate);
		this.hiddenBiasNeurons = this.hiddenBiasNeurons.plus( hBiasAdd.times( learningRate ) );
		
		
		
	}
/*	
	public void setLearningRate(double alpha) {
		
		this.learningRate = alpha;
		
	}
*/	
	/**
	 * Used to calculate how well trained the RBM currently is
	 * 
	 * [ TODO: Currently giving weird numbers ]
	 * 
	 * - the input training set matrix is arranged such that there is a training example per row
	 * 
	 * @return
	 */
	public double getReConstructionCrossEntropy() {
		
/*		
 * 
 cross_entropy = T.mean(
                	T.sum(
                	
                		self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) 
                		+
                		(1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                    
                      axis=1)
                 )
 * 
	*/	
		
		// 1. get sigmoid of the inputMatrix x weights
		
		// probably could just call the propUp call w the training dataset as param
		Matrix preSigmoidHidden = this.trainingDataset.times( this.connectionWeights );
		preSigmoidHidden = MatrixUtils.addRowVector(preSigmoidHidden, this.hiddenBiasNeurons.viewRow(0));
		Matrix sigHidden = MatrixUtils.sigmoid(preSigmoidHidden);
		
		
		
		// 2. get sigmoid of the ( sigH from #1 x transpose(weights) )
		
		// could use propDown here
		Matrix preSigmoidVis = sigHidden.times( this.connectionWeights.transpose() );
		preSigmoidVis = MatrixUtils.addRowVector(preSigmoidVis, this.visibleBiasNeurons.viewRow(0));
		Matrix sigVis = MatrixUtils.sigmoid( preSigmoidVis );
		
		
		// 3. put together the partials to build the cross entropy
		
		Matrix logSigmoidVis = MatrixUtils.log(sigVis);
		Matrix oneMinusSigmoidVis = MatrixUtils.oneMinus(sigVis); //MatrixUtils.ones(sigVis.numRows(), sigVis.numCols()).minus(sigVis);


		
		Matrix oneMinusInput = MatrixUtils.oneMinus(this.trainingDataset);
		
		
		Matrix logOneMinusSigVisible = MatrixUtils.log(oneMinusSigmoidVis);
	
//		MatrixUtils.debug_print_matrix_stats(this.trainingDataset, "training dataset");
//		MatrixUtils.debug_print_matrix_stats( logSigmoidVis, "logSigV");
		
		
		// D
		// self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv))
		Matrix inputTimesLogSigVisible = MatrixUtils.elementWiseMultiplication(this.trainingDataset,logSigmoidVis);
		
		// E
		// (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv))
		Matrix oneMinusInputTimesLogOneMinusSigV = MatrixUtils.elementWiseMultiplication(oneMinusInput,logOneMinusSigVisible);
		
		// D + E
		Matrix Inner = inputTimesLogSigVisible.plus(oneMinusInputTimesLogOneMinusSigV);

		// get the sum for each row (example)
		Matrix crossEntRowSums = MatrixUtils.rowSums(Inner);
		
		//MatrixUtils.debug_print(crossEntRowSums);
		
		// now compute the average across the cross-entropies for each training example
		double crossEntFinal = MatrixUtils.mean(crossEntRowSums);
				
		return -crossEntFinal;
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
	//public Matrix generateProbabilitiesForHiddenStatesBasedOnVisibleStates(Matrix visible) {
	public Matrix propUp(Matrix visible) {
		
		Matrix preSigmoid = visible.times( this.connectionWeights );
		preSigmoid = MatrixUtils.addRowVector(preSigmoid, this.hiddenBiasNeurons.viewRow(0));

		return MatrixUtils.sigmoid(preSigmoid);
	}

	/**
	 * Binomial sampling of the hidden values given visible
	 * 
	 * This function infers state of hidden units given visible units
	 * 
	 * Used in Gibbs Sampling
	 * 
	 * @param visible
	 */
	public Pair<Matrix, Matrix> sampleHiddenGivenVisible(Matrix visible) {
				
		//Matrix hiddenProbs = this.generateProbabilitiesForHiddenStatesBasedOnVisibleStates(visible);
		Matrix hiddenProbs = this.propUp(visible);

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
	//public Matrix generateProbabilitiesForVisibleStatesBasedOnHiddenStates(Matrix hidden) {
	public Matrix propDown(Matrix hidden) {
		
		Matrix preSigmoid = hidden.times( this.connectionWeights.transpose() );
		preSigmoid = MatrixUtils.addRowVector(preSigmoid, this.visibleBiasNeurons.viewRow(0));

		return MatrixUtils.sigmoid(preSigmoid);
	}
	
	/**
	 * This function infers state of visible units given hidden units
	 * 
	 * Used in Gibbs Sampling
	 * 
	 * @param hidden
	 */
	public Pair<Matrix, Matrix> sampleVisibleGivenHidden(Matrix hidden) {
		
		//Matrix visibleProb = this.generateProbabilitiesForVisibleStatesBasedOnHiddenStates(hidden);
		Matrix visibleProb = this.propDown(hidden);

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
	
	/**
	 * 
	 * Reconstructs the visible input.
	 * A reconstruction is a propagation down of the reconstructed hidden input.
	 * 
	 * TODO: this is a duplicate method, track down refs and remove
	 * 
	 */
	public Matrix reconstructVisibleInput(Matrix visible) {

		// propUp
//		Matrix propUpHiddenResult = this.generateProbabilitiesForHiddenStatesBasedOnVisibleStates(visible);
		Matrix propUpHiddenResult = this.propUp(visible);

		
		//return propDown(h);
//		return this.generateProbabilitiesForVisibleStatesBasedOnHiddenStates(propUpHiddenResult);
		return this.propDown(propUpHiddenResult);
	}

	
	/**
	 * TODO: Think about the object[] array as parameter mechanism a bit more
	 * 
	 */
	@Override
	public void trainTillConvergence(Matrix input, double learningRate, Object[] params) {
		// still java 6!
		int k = (Integer) params[0];
		trainTillConvergence(learningRate, k, input);
	}
	
	/**
	 * Trains till global minimum is found.
	 * 
	 * TODO: implement the newer style optimizer stuff here!
	 * 
	 * @param learningRate
	 * @param k
	 * @param input
	 */
	public void trainTillConvergence(double learningRate, int k, Matrix input) {
		if (input != null) {
			this.trainingDataset = input;
		}
		//this.learningRate = learningRate;
		
		//optimizer = new RBMOptimizer(this, learningRate, new Object[]{k});
		//optimizer.train(input);
		
		optimizer = new RestrictedBoltzmannMachineOptimizer(this, learningRate, new Object[]{k});
		optimizer.train(input);

		
	}
	

	@Override
	public double lossFunction(Object[] params) {
		return getReConstructionCrossEntropy();
	}

	@Override
	public void train(Matrix input, double learningRate, Object[] params) {
		int k = (Integer) params[0];
		//this.learningRate = learningRate;
		contrastiveDivergence(learningRate, k, input);
	}

	/**
	 * Reconstructs the visible input.
	 * A reconstruction is a propdown of the reconstructed hidden input.
	 * @param v the visible input
	 * @return the reconstruction of the visible input
	 */
	@Override
	public Matrix reconstruct(Matrix v) {
	
		return propDown(propUp(v));
		
	}
	
	
	
}
