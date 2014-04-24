package tv.floe.metronome.deeplearning.rbm;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;



import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseNeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.gradient.NeuralNetworkGradient;
import tv.floe.metronome.deeplearning.neuralnetwork.layer.HiddenLayer;
import tv.floe.metronome.deeplearning.neuralnetwork.optimize.NeuralNetworkOptimizer;
import tv.floe.metronome.math.MathUtils;
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
	public transient NeuralNetworkOptimizer optimizer;
	public double[] debugWeightAddsBuffer = null; // only set when we want to check things
	
	public RestrictedBoltzmannMachine() { }

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
		
		
		
		NormalDistribution u = new NormalDistribution(this.randNumGenerator,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);

		this.connectionWeights = new DenseMatrix( this.numberVisibleNeurons, this.numberHiddenNeurons );
		this.connectionWeights.assign(0.0);
		
		for (int r = 0; r < this.connectionWeights.numRows(); r++) {
			
			for(int c = 0; c < this.connectionWeights.numCols(); c++) { 
			
				double init_weight = u.sample( );
				
			//	System.out.println("w: " + init_weight);
				
				this.connectionWeights.setQuick( r, c, init_weight );
			
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

		//System.out.println("Creating RBM: visible: " + this.numberVisibleNeurons + ", hidden: " + this.numberHiddenNeurons );
		
		
	}
	
	
	public void setupCDkDebugBuffer(double[] buffer) {
		this.debugWeightAddsBuffer = buffer;
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
	 * 
	 * @param k
	 */
	public void contrastiveDivergence(double learningRate, int k, Matrix input) {

		if (input != null) {
			
			this.trainingDataset = input;
			
		}
		
		// actually performs CDk
		NeuralNetworkGradient gradient = getGradient(new Object[]{k,learningRate});
		
		//W.addi(gradient.getwGradient());
		this.connectionWeights = this.connectionWeights.plus(gradient.getwGradient());
		
		//hBias.addi(gradient.gethBiasGradient());
		this.hiddenBiasNeurons = this.hiddenBiasNeurons.plus(gradient.gethBiasGradient());
		
		//vBias.addi(gradient.getvBiasGradient());
		this.visibleBiasNeurons = this.visibleBiasNeurons.plus( gradient.getvBiasGradient() );
		

	}
	

	
	/**
	 * reviewed, seems decent
	 * 
	 */
	@Override
	public NeuralNetworkGradient getGradient(Object[] params) {
		int k = (Integer) params[0];
		double learningRate = (Double) params[1];
		
		// init CDk
		
		// do gibbs sampling given V to get the Hidden states based on the training input
		// compute positive phase
		Pair<Matrix, Matrix> hiddenProbsAndSamplesStart = this.sampleHiddenGivenVisible( this.trainingDataset );
				
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
		Matrix trainingDataTimesInitialHiddenStates = this.trainingDataset.transpose().times( hiddenProbsAndSamplesStart.getSecond() );

		// now compute the <vi hj>model (this may be vi * phj --- double check)
		Matrix negativeVisibleSamplesTransposeTimesNegHiddenExpValues = negativeVisibleSamples.transpose().times( negativeHiddenExpectedValues );
				
		// calc the delta between: data - model
		Matrix dataModelDelta = trainingDataTimesInitialHiddenStates.minus( negativeVisibleSamplesTransposeTimesNegHiddenExpValues );
		
		// learningRate * delta(data - model)
		Matrix wGradient = dataModelDelta.times( learningRate );
		
		if (useRegularization) { 
		
			//wGradient.subi(W.muli(l2));
			
			//System.out.println("regularization! " + l2);
			
			// TODO: figure out if this should stick around like this
			this.connectionWeights = this.connectionWeights.times(l2);
			wGradient = wGradient.minus(this.connectionWeights);
		
		}

		
		if (momentum != 0) {
		
			//wGradient.muli( 1 - momentum);
			
			wGradient = wGradient.times( 1 - momentum );
			
		}

		
		// This is added normalization for mini-batching
		
		//wGradient.divi(input.rows);
//		wGradient = wGradient.divide( this.trainingDataset.numRows() );
		
		
		// ---- end of equation (9) section -----------------
		
		// update the connection weights and bias terms for visible/hidden units
		//this.connectionWeights = this.connectionWeights.plus( connectionWeightChanges );

		Matrix vBiasGradient = MatrixUtils.mean( this.trainingDataset.minus( negativeVisibleSamples ) , 0).times( learningRate ); 
		//this.visibleBiasNeurons = this.visibleBiasNeurons.plus( vBiasAdd );

		Matrix hBiasGradient = null;
		
		if(this.sparsity != 0) {
			//all hidden units must stay around this number
//			hBiasGradient = mean(probHidden.getSecond().add( -sparsity),0).mul(learningRate);
			hBiasGradient = MatrixUtils.mean( hiddenProbsAndSamplesStart.getSecond().plus( -sparsity ) , 0).times( learningRate ); //.times(this.learningRate);
		}
		else {
			//update rule: the expected values of the hidden input - the negative hidden  means adjusted by the learning rate
//			hBiasGradient = mean(probHidden.getSecond().sub(nhMeans), 0).mul(learningRate);
			hBiasGradient = MatrixUtils.mean( hiddenProbsAndSamplesStart.getSecond().minus( negativeHiddenExpectedValues ) , 0).times( learningRate ); //.times(this.learningRate);
		}
		
		
		
		//this.hiddenBiasNeurons = this.hiddenBiasNeurons.plus( hBiasAdd );		

		return new NeuralNetworkGradient(wGradient, vBiasGradient, hBiasGradient);
		
		
	}

	/**
	 * Reconstruction entropy.
	 * 
	 * Used to calculate how well trained the RBM currently is
	 * 
	 * - the input training set matrix is arranged such that there is a training example per row
	 * 
	 * This compares the similarity of two probability
	 * distributions, in this case that would be the input
	 * and the reconstructed input with gaussian noise.
	 * This will account for either regularization or none
	 * depending on the configuration.
	 * 
	 * @return reconstruction error
	 */
	public double getReConstructionCrossEntropy() {

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
				
		Matrix hiddenProbs = this.propUp(visible);

		Matrix hiddenBinomialSamples = MatrixUtils.genBinomialDistribution(hiddenProbs, 1, this.randNumGenerator);
		
		return new Pair<Matrix, Matrix>(hiddenProbs, hiddenBinomialSamples);
	}
	
	
	/**
	 * 
	 * 
	 * @param visible
	 * @return
	 */
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

	/**
	 * Free energy for an RBM
	 * Lower energy models have higher probability
	 * of activations
	 * @param visibleSample the sample to test on
	 * @return the free engery for this sample
	 */
	public double freeEnergy(Matrix visibleSample) {

		Matrix wxB = MatrixUtils.addRowVector( visibleSample.times( this.connectionWeights ), this.hiddenBiasNeurons.viewRow( 0 ) );
		
		double vBiasTerm = MathUtils.dot(visibleSample, this.visibleBiasNeurons);
		
		double hBiasTerm = MatrixUtils.sum( MatrixUtils.log( MatrixUtils.exp( wxB ).plus( 1.0 ) ) );
		
		return -hBiasTerm - vBiasTerm;
		
	}
	
	/**
	 * 
	 * Reconstructs the visible input.
	 * A reconstruction is a propagation down of the reconstructed hidden input.
	 * 
	 * 
	 */
	public Matrix reconstructVisibleInput(Matrix visible) {

		// propUp
		Matrix propUpHiddenResult = this.propUp(visible);
		
		return this.propDown(propUpHiddenResult);
	}

	
	/**
	 * TODO: Think about the object[] array as parameter mechanism a bit more
	 * 
	 */
	@Override
	public void trainTillConvergence(Matrix input, double learningRate, Object[] params) {
		
		if ( null != input) {
			this.trainingDataset = input;
		}
		
		// still java 6!
		//int k = (Integer) params[0];
		//trainTillConvergence(learningRate, k, input);
		
		optimizer = new RestrictedBoltzmannMachineOptimizer( this, learningRate, params );
		optimizer.train( input );
		
		
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

		//System.out.println("using Mallet's optimization");
		
		optimizer = new RestrictedBoltzmannMachineOptimizer(this, learningRate, new Object[]{ k });
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
	
	/*
	 * 

		
	 * 
	 */

	/**
	 * Copies params from the passed in network
	 * to this one
	 * @param n the network to copy
	 */
	public void update(BaseNeuralNetworkVectorized n) {
		/*
		this.W = n.W;
		this.hBias = n.hBias;
		this.vBias = n.vBias;
		this.l2 = n.l2;
		this.useRegularization = n.useRegularization;
		this.momentum = n.momentum;
		this.nHidden = n.nHidden;
		this.nVisible = n.nVisible;
		this.rng = n.rng;
		this.sparsity = n.sparsity;
		*/
	}


	/**
	 * Serializes this to the output stream.
	 * @param os the output stream to write to
	 */
	public void write(OutputStream os) {
		try {

		    DataOutput d = new DataOutputStream(os);
		    ObjectOutputStream oos = new ObjectOutputStream(os);
		    
		    d.writeInt( this.numberVisibleNeurons );
		    d.writeInt( this.numberHiddenNeurons );
		    
		    MatrixWritable.writeMatrix(d, this.hiddenBiasNeurons );
		    MatrixWritable.writeMatrix(d, this.visibleBiasNeurons );
		    MatrixWritable.writeMatrix(d, this.connectionWeights );
//		    MatrixWritable.writeMatrix(d, this.trainingDataset );	
		    		    
		    oos.writeObject( this.randNumGenerator );

			d.writeDouble( this.sparsity ); 
			d.writeDouble( this.momentum );
			d.writeDouble( this.l2 );
			d.writeInt( this.renderWeightsEveryNumEpochs );
			d.writeDouble( this.fanIn );
			d.writeBoolean( this.useRegularization );

		    

		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}	
	
	
	
	
	/**
	 * Load (using {@link ObjectInputStream}
	 * @param is the input stream to load from (usually a file)
	 */
	public void load(InputStream is) {
		try {

			DataInput di = new DataInputStream(is);
			
			//this.nIn = di.readInt();
//			this.input = MatrixWritable.readMatrix( di );


		    //DataOutput d = new DataOutputStream(os);
		    ObjectInputStream ois = new ObjectInputStream(is);
		    
		    this.numberVisibleNeurons = di.readInt();
		    this.numberHiddenNeurons = di.readInt();
		    
		    this.hiddenBiasNeurons = MatrixWritable.readMatrix( di );
		    this.visibleBiasNeurons = MatrixWritable.readMatrix( di );
		    this.connectionWeights = MatrixWritable.readMatrix( di );
//		    this.trainingDataset = MatrixWritable.readMatrix( di );	
		    		    
		    this.randNumGenerator = (RandomGenerator) ois.readObject();

			this.sparsity = di.readDouble(); 
			this.momentum = di.readDouble();
			this.l2 = di.readDouble();
			this.renderWeightsEveryNumEpochs = di.readInt();
			this.fanIn = di.readDouble();
			this.useRegularization = di.readBoolean();
			
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}	
	
	/**
	 * Serializes this to the output stream.
	 * 
	 * Used in parameter averaging
	 * 
	 * @param os the output stream to write to
	 */
	public void serializeParameters(OutputStream os) {
		try {

		    DataOutput d = new DataOutputStream(os);
		    
		    d.writeInt( this.numberVisibleNeurons );
		    d.writeInt( this.numberHiddenNeurons );
		    
		    // ??
		    MatrixWritable.writeMatrix(d, this.hiddenBiasNeurons );
		    // ??
		    MatrixWritable.writeMatrix(d, this.visibleBiasNeurons );
		    // yes
		    MatrixWritable.writeMatrix(d, this.connectionWeights );
		    		    
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}		
	
	/**
	 * Load parameter values from the byte stream 
	 * 
	 */
	public void loadParameterValues(InputStream is) {
		try {

			DataInput di = new DataInputStream(is);
			
		    this.numberVisibleNeurons = di.readInt();
		    this.numberHiddenNeurons = di.readInt();
		    
		    this.hiddenBiasNeurons = MatrixWritable.readMatrix( di );
		    this.visibleBiasNeurons = MatrixWritable.readMatrix( di );
		    this.connectionWeights = MatrixWritable.readMatrix( di );
		    				
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}		
	
	
}
