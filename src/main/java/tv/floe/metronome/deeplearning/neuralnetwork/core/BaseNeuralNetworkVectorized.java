package tv.floe.metronome.deeplearning.neuralnetwork.core;


import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.neuralnetwork.layer.HiddenLayer;
import tv.floe.metronome.math.MatrixUtils;

/**
 * Based on the NN design by Adam Gibson
 * 
 * 
 * 
 * Connections are based on a concept of a matrix  of size ( visibleNeurons x hiddenNeurons )
 * - where each row represents the connections for a specific visible neuron (n-th) to all other hidden neurons (m-th) 
 * 
 * Hidden and Visible Neurons
 * - these are 1 x NeuronCount dim Matrices (vectors, really) that hold the states of each visible or hidden neuron based on activations
 * 
 * @author josh
 *
 */
public abstract class BaseNeuralNetworkVectorized implements NeuralNetworkVectorized {

	//public int inputNeuronCount;
	
	public int numberVisibleNeurons;
	public int numberHiddenNeurons;
	
	public Matrix hiddenBiasNeurons;
	public Matrix visibleBiasNeurons;
	public Matrix connectionWeights;
	public Matrix trainingDataset = null;	
		
	public RandomGenerator randNumGenerator;
	

	public double sparsity = 0.01;
	/* momentum for learning */
	public double momentum = 0.1;
	/* L2 Regularization constant */
	public double l2 = 0.1;

	public int renderWeightsEveryNumEpochs = -1;
	
	public double fanIn = -1;
	public boolean useRegularization = true;
	
	
	// default CTOR
	public BaseNeuralNetworkVectorized() {
		
	}
	
	public BaseNeuralNetworkVectorized(int nVisible, int nHidden, Matrix weights, Matrix hBias, Matrix vBias, RandomGenerator rng) {
		this.numberVisibleNeurons = nVisible;
		this.numberHiddenNeurons = nHidden;

		if (rng == null)	{
			this.randNumGenerator = new MersenneTwister(1234);
		} else { 
			this.randNumGenerator = rng;
		}

		if (weights == null) {
			
			double a = 1.0 / (double) nVisible;
			/*
			 * Initialize based on the number of visible units..
			 * The lower bound is called the fan in
			 * The outer bound is called the fan out.
			 * 
			 * Below's advice works for Denoising AutoEncoders and other 
			 * neural networks you will use due to the same baseline guiding principles for
			 * both RBMs and Denoising Autoencoders.
			 * 
			 * Hinton's Guide to practical RBMs:
			 * The weights are typically initialized to small random values chosen from a zero-mean Gaussian with
			 * a standard deviation of about 0.01. Using larger random values can speed the initial learning, but
			 * it may lead to a slightly worse final model. Care should be taken to ensure that the initial weight
			 * values do not allow typical visible vectors to drive the hidden unit probabilities very close to 1 or 0
			 * as this significantly slows the learning.
			 */
			NormalDistribution u = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);

			//this.connectionWeights = Matrix.zeros(nVisible,nHidden);
			this.connectionWeights = new DenseMatrix( nVisible, nHidden );
			this.connectionWeights.assign(0);

			for(int i = 0; i < this.connectionWeights.numRows(); i++) {
				for(int j = 0; j < this.connectionWeights.numCols(); j++) {
					this.connectionWeights.set(i,j,u.sample());
				}
			}


		} else {	
			this.connectionWeights = weights;
		}


		if (hBias == null) { 
			// TODO: recheck if this column vector is correctly oriented
			this.hiddenBiasNeurons = new DenseMatrix(1, nHidden); //Matrix.zeros(nHidden);
		//} else if(hBias.numRows() != nHidden) {
			//throw new IllegalArgumentException("Hidden bias must have a length of " + nHidden + " length was " + hBias.numRows());
		} else {
			this.hiddenBiasNeurons = hBias;
		}

		if (vBias == null) { 
			this.visibleBiasNeurons = new DenseMatrix(1, nVisible); //Matrix.zeros(nVisible);
			this.visibleBiasNeurons.assign(0);

		} else if(vBias.numRows() != nVisible) { 
			throw new IllegalArgumentException("Visible bias must have a length of " + nVisible + " but length was " + vBias.numRows());

		} else { 
			this.visibleBiasNeurons = vBias;
		}
	}	
	
	public BaseNeuralNetworkVectorized(Matrix input, int nVisible, int nHidden, Matrix weights, Matrix hBias, Matrix vBias, RandomGenerator rng) {

		this(nVisible, nHidden, weights, hBias, vBias, rng);
		this.trainingDataset = input;
	}
	
	
	@Override
	public int getnVisible() {

		return this.numberVisibleNeurons;
		
	}

	@Override
	public void setnVisible(int nVisible) {

		this.numberVisibleNeurons = nVisible;
		
	}

	@Override
	public int getnHidden() {
		
		return this.numberHiddenNeurons;
	}

	@Override
	public void setnHidden(int nHidden) {

		this.numberHiddenNeurons = nHidden;
		
	}

	@Override
	public Matrix getConnectionWeights() {

		return this.connectionWeights;
		
	}

	@Override
	public void setConnectionWeights(Matrix weights) {

		this.connectionWeights = weights;
		
	}

	@Override
	public Matrix getHiddenBias() {

		return this.hiddenBiasNeurons;
		
	}

	@Override
	public void sethBias(Matrix hiddenBias) {
		
		this.hiddenBiasNeurons = hiddenBias;
		
	}

	@Override
	public Matrix getVisibleBias() {

		return this.visibleBiasNeurons;
		
	}

	@Override
	public void setVisibleBias(Matrix visibleBias) {

		this.visibleBiasNeurons = visibleBias;
		
	}

	@Override
	public RandomGenerator getRng() {

		return this.randNumGenerator;
		
	}

	@Override
	public void setRng(RandomGenerator rng) {

		this.randNumGenerator = rng;
		
	}

	@Override
	public Matrix getInput() {
		// TODO Auto-generated method stub
		return this.trainingDataset;
	}

	@Override
	public void setInput(Matrix input) {

		this.trainingDataset = input;
		
	}
	
	@Override
	public NeuralNetworkVectorized transpose() {
		try {
			NeuralNetworkVectorized ret = getClass().newInstance();
			ret.sethBias( this.hiddenBiasNeurons.clone() );
			ret.setVisibleBias( this.visibleBiasNeurons.clone() );
			ret.setnHidden(getnVisible());
			ret.setnVisible(getnHidden());
			ret.setConnectionWeights( this.connectionWeights.transpose() );
			ret.setRng(getRng());

			return ret;
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 


	}

	@Override
	public NeuralNetworkVectorized clone() {
		try {
			NeuralNetworkVectorized ret = getClass().newInstance();
			ret.sethBias( this.hiddenBiasNeurons.clone() );
			ret.setVisibleBias( this.visibleBiasNeurons.clone() );
			ret.setnHidden(getnHidden());
			ret.setnVisible(getnVisible());
			//ret.setW(W.dup());
			ret.setConnectionWeights( this.connectionWeights.clone() );
			ret.setRng(getRng());

			return ret;
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 


	}
	

	@Override
	public double getSparsity() {
		return this.sparsity;
	}

	@Override
	public void setSparsity(double sparsity) {

		this.sparsity = sparsity;
	}

	@Override
	public double getL2() {
		return this.l2;
	}

	@Override
	public void setL2(double l2) {

		this.l2 = l2;
		
	}

	@Override
	public double getMomentum() {
		
		return this.momentum;
		
	}

	@Override
	public void setMomentum(double momentum) {

		this.momentum = momentum;
		
	}

/*	@Override
	public void trainTillConvergence(Matrix input, double lr, Object[] params) {
		// TODO Auto-generated method stub
		
	}	
	*/
	
	/**
	 * All neural networks are based on this idea of 
	 * minimizing reconstruction error.
	 * Both RBMs and Denoising AutoEncoders
	 * have a component for reconstructing, ala different implementations.
	 *  
	 * @param x the input to reconstruct
	 * @return the reconstructed input
	 */
	public abstract Matrix reconstruct(Matrix x);
	
	/**
	 * The loss function (cross entropy, reconstruction error,...)
	 * @return the loss function
	 */
	public abstract double lossFunction(Object[] params);

	public double lossFunction() {
		return lossFunction(null);
	}
	
	@Override
	public double squaredLoss() {
		Matrix reconstructed = reconstruct( this.trainingDataset );
		
		//double loss = MatrixFunctions.powi(reconstructed.sub(input), 2).sum() / input.rows;
		double loss = MatrixUtils.sum( MatrixUtils.pow( reconstructed.minus( this.trainingDataset ), 2 ) ) / this.trainingDataset.numRows();
		
		if(this.useRegularization) {
			//loss += 0.5 * l2 * MatrixFunctions.pow(W,2).sum();
			loss += 0.5 * l2 * MatrixUtils.sum( MatrixUtils.pow( this.connectionWeights, 2 ) );
		}
		
		return -loss;
	}
	
	
	/**
	 * Train one iteration of the network
	 * @param input the input to train on
	 * @param lr the learning rate to train at
	 * @param params the extra params (k, corruption level,...)
	 */
	public abstract void train(Matrix input, double learningRate, Object[] params);
		
	
	
	
	/**
	 * Reconstruction error.
	 * 
	 * Reconstruction entropy.
	 * This compares the similarity of two probability
	 * distributions, in this case that would be the input
	 * and the reconstructed input with gaussian noise.
	 * This will account for either regularization or none
	 * depending on the configuration.
	 * 
	 * @return reconstruction error
	 */
	public double getReConstructionCrossEntropy() {
		//Matrix preSigH = input.mmul(W).addRowVector(hBias);
		Matrix preSigH = MatrixUtils.addRowVector( this.trainingDataset.times(this.connectionWeights), this.hiddenBiasNeurons.viewRow(0) );
		Matrix sigH = MatrixUtils.sigmoid(preSigH);

		Matrix preSigV = MatrixUtils.addRowVector( sigH.times(this.connectionWeights.transpose()), this.visibleBiasNeurons.viewRow(0) );
		Matrix sigV = MatrixUtils.sigmoid(preSigV);
		Matrix inner = 
				this.trainingDataset.times(MatrixUtils.log(sigV))
				.plus(MatrixUtils.oneMinus( this.trainingDataset )
						.times(MatrixUtils.log(MatrixUtils.oneMinus(sigV))));
		
		//double l = inner.length;
		double l = MatrixUtils.length(inner);
		
		if (this.useRegularization) {
			double normalized = l + l2RegularizedCoefficient();
			return - MatrixUtils.mean( MatrixUtils.rowSums( inner ) ) / normalized;
		}
		
		return - MatrixUtils.mean( MatrixUtils.rowSums( inner ) );
	}	
	
	@Override
	public double l2RegularizedCoefficient() {
		//return (MatrixFunctions.pow(getW(),2).sum()/ 2.0)  * l2;
		
		return ( MatrixUtils.sum( MatrixUtils.pow( this.getConnectionWeights(), 2 ) ) / 2.0 ) * l2;
		
	}
	
	protected void initWeights()  {
		
		if (this.numberVisibleNeurons < 1) {
			throw new IllegalStateException("Number of visible can not be less than 1");
		}
		
		if (this.numberHiddenNeurons < 1) {
			throw new IllegalStateException("Number of hidden can not be less than 1");
		}
		
		
		/*
		 * Initialize based on the number of visible units..
		 * The lower bound is called the fan in
		 * The outer bound is called the fan out.
		 * 
		 * Below's advice works for Denoising AutoEncoders and other 
		 * neural networks you will use due to the same baseline guiding principles for
		 * both RBMs and Denoising Autoencoders.
		 * 
		 * Hinton's Guide to practical RBMs:
		 * The weights are typically initialized to small random values chosen from a zero-mean Gaussian with
		 * a standard deviation of about 0.01. Using larger random values can speed the initial learning, but
		 * it may lead to a slightly worse final model. Care should be taken to ensure that the initial weight
		 * values do not allow typical visible vectors to drive the hidden unit probabilities very close to 1 or 0
		 * as this significantly slows the learning.
		 */
		
		if (this.connectionWeights == null) {
			
			NormalDistribution u = new NormalDistribution( this.randNumGenerator, 0, .01, NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY );

			this.connectionWeights = new DenseMatrix( this.numberVisibleNeurons, this.numberHiddenNeurons );// Matrix.zeros(nVisible,nHidden);
			this.connectionWeights.assign(0.0);

		//	for(int i = 0; i < this.W.rows; i++) 
		//		this.W.putRow(i,new Matrix(u.sample(this.W.columns)));
			
			for ( int i = 0; i < this.connectionWeights.numRows(); i++ ) {
				
				// u.sample( "number of cols in weights" )
				
				double[] rowSamples = u.sample( this.connectionWeights.numCols() );
				
				this.connectionWeights.viewRow(i).assign(rowSamples);
				
				
			}

		}

		//if(this.hBias == null) {
		if ( this.hiddenBiasNeurons == null) {
			
			//this.hBias = Matrix.zeros(nHidden);
			this.hiddenBiasNeurons = new DenseMatrix(1, this.numberHiddenNeurons);// Matrix.zeros(nHidden);
			this.hiddenBiasNeurons.assign(0.0);
			
			/*
			 * Encourage sparsity.
			 * See Hinton's Practical guide to RBMs
			 */
			//this.hBias.subi(4);
		}

		if (this.visibleBiasNeurons == null) {
			
			if (this.trainingDataset != null) {
		
				this.visibleBiasNeurons = new DenseMatrix(1, this.numberVisibleNeurons); // Matrix.zeros(nVisible);
				this.visibleBiasNeurons.assign(0.0);


			} else {
//				this.vBias = Matrix.zeros(nVisible);
				this.visibleBiasNeurons = new DenseMatrix(1, this.numberVisibleNeurons); // Matrix.zeros(nVisible);
				this.visibleBiasNeurons.assign(0.0);
				
			}
		}



	}
	
	@Override
	public void clearWeights() {
		
	//	this.connectionWeights = new DenseMatrix( this.numberVisibleNeurons, this.numberHiddenNeurons );// Matrix.zeros(nVisible,nHidden);
		this.connectionWeights.assign(0.0);
		
	//	this.hiddenBiasNeurons = new DenseMatrix(1, this.numberHiddenNeurons);// Matrix.zeros(nHidden);
		this.hiddenBiasNeurons.assign(0.0);
		
	//	this.visibleBiasNeurons = new DenseMatrix(1, this.numberVisibleNeurons); // Matrix.zeros(nVisible);
		this.visibleBiasNeurons.assign(0.0);
		
		
	}


	@Override
	public void setRenderEpochs(int renderEpochs) {
		this.renderWeightsEveryNumEpochs = renderEpochs;

	}
	@Override
	public int getRenderEpochs() {
		return renderWeightsEveryNumEpochs;
	}

	@Override
	public double fanIn() {
		return fanIn < 0 ? 1 / this.numberVisibleNeurons : fanIn;
	}

	@Override
	public void setFanIn(double fanIn) {
		this.fanIn = fanIn;
	}

	public void regularize() {
	
		//this.W.addi(W.mul(0.01));
		//this.W.divi(this.momentum);
		
		this.connectionWeights = this.connectionWeights.plus( this.connectionWeights.times(0.01) );
		this.connectionWeights = this.connectionWeights.divide( this.momentum );

	}

	public void scaleWeights( double scale ) {
		
		this.connectionWeights = this.connectionWeights.times( scale );
		
	}
	
	/**
	 * Performs a (parameter average) network merge in the form of
	 * a += b - a / n
	 * where a is a matrix here
	 * b is a matrix on the incoming network
	 * and n is the batch size
	 * @param network the network to merge with
	 * @param batchSize the batch size (number of training examples)
	 * to average by
	 */	
	@Override
	public void merge(NeuralNetworkVectorized network,int batchSize) {

		//			W.addi(network.getW().mini(W).div(batchSize));
		
		MatrixUtils.addi( this.connectionWeights, ( network.getConnectionWeights().minus( this.connectionWeights ).divide(batchSize) ) );
		
//			hBias.addi(network.gethBias().subi(hBias).divi(batchSize));

		MatrixUtils.addi( this.hiddenBiasNeurons, ( network.getHiddenBias().minus( this.hiddenBiasNeurons ).divide( batchSize ) ) );
		
		//			vBias.addi(network.getvBias().subi(vBias).divi(batchSize));

		MatrixUtils.addi( this.visibleBiasNeurons, ( network.getVisibleBias().minus( this.visibleBiasNeurons ).divide( batchSize ) ) );
		
	}
	
	public void jostleWeighMatrix() {
		/*
		 * Initialize based on the number of visible units..
		 * The lower bound is called the fan in
		 * The outer bound is called the fan out.
		 * 
		 * Below's advice works for Denoising AutoEncoders and other 
		 * neural networks you will use due to the same baseline guiding principles for
		 * both RBMs and Denoising Autoencoders.
		 * 
		 * Hinton's Guide to practical RBMs:
		 * The weights are typically initialized to small random values chosen from a zero-mean Gaussian with
		 * a standard deviation of about 0.01. Using larger random values can speed the initial learning, but
		 * it may lead to a slightly worse final model. Care should be taken to ensure that the initial weight
		 * values do not allow typical visible vectors to drive the hidden unit probabilities very close to 1 or 0
		 * as this significantly slows the learning.
		 */
		NormalDistribution u = new NormalDistribution( this.randNumGenerator, 0, .01, fanIn() );

		Matrix weights = new DenseMatrix( this.numberVisibleNeurons, this.numberHiddenNeurons ); //Matrix.zeros(nVisible,nHidden);
		weights.assign(0.0);

		for (int i = 0; i < this.connectionWeights.numRows(); i++) { 
		
			// TODO: figure out whats going on with the weights matrix
		//	weights.putRow(i,new Matrix(u.sample(this.W.columns)));
			
		}




	}	
	
	
}
