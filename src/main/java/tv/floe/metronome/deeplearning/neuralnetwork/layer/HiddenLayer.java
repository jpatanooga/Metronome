package tv.floe.metronome.deeplearning.neuralnetwork.layer;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import tv.floe.metronome.math.MatrixUtils;

public class HiddenLayer {

	
	private int neuronCountPreviousLayer = 0;
	private int neuronCount = 0;
	
	public Matrix connectionWeights;
	public Matrix biasTerms;
	public RandomGenerator rndNumGenerator;
	public Matrix input;
	//private ActivationFunction sigmoidFunction = new Sigmoid();
	
	/**
	 * LayerVectorized Ctor
	 * 
	 * we use the count of neurons of the prev layer and this layer's neuron count to deduce the weight matrix
	 * 
	 * @param neuronCountPrevLayer
	 * @param neuronCount
	 * @param rndGen
	 */
	public HiddenLayer(int neuronCountPrevLayer, int neuronCount, RandomGenerator rndGen) {
		
		this.neuronCount = neuronCount;
		this.neuronCountPreviousLayer = neuronCountPrevLayer;
		this.rndNumGenerator = rndGen;
		
		double a = 1.0 / (double) neuronCountPrevLayer;

		UniformRealDistribution realDistributionGenerator = new UniformRealDistribution( this.rndNumGenerator, -a, a );

		
		// init the connection weights
		this.connectionWeights = new DenseMatrix( this.neuronCountPreviousLayer, this.neuronCount );
		this.connectionWeights.assign(0.0);

		for (int r = 0; r < this.connectionWeights.numRows(); r++) {
			
			for(int c = 0; c < this.connectionWeights.numCols(); c++) { 
			
				this.connectionWeights.setQuick( r, c, realDistributionGenerator.sample() );
			
			}

		}	
		
		// init the bias terms (column vector)
		// what operations use this Matrix / column vector?
		this.biasTerms = new DenseMatrix( 1, this.neuronCount );
		this.biasTerms.assign(0.0);
		
		
	}
	
	/**
	 * In the vectorized implementation we want to hook the previous layer's
	 * work in progress into this layer
	 * 
	 * In the case of the first layer, its just the matrix of input samples
	 * 
	 * @param layerInput
	 */
	public void setInput(Matrix layerInput) {
		this.input = layerInput;
	}
	
	public void setWeights(Matrix weights) {
		
		this.connectionWeights = weights;
		
	}
	
	
	public Matrix getVectorizedOutputMatrix() {
		
		return null;
	}
	
	
	/**
	 * Trigger an activation with the last specified input
	 * @return the activation of the last specified input
	 */
	public Matrix computeActivationOutput() {
		
		Matrix mult = this.input.times(connectionWeights);
		Matrix multPlusBias = MatrixUtils.addRowVector(mult, this.biasTerms.viewRow(0));
		return MatrixUtils.sigmoid(multPlusBias);

	}
	
	
	
	/**
	 * Initialize the layer with the given input
	 * and return the activation for this layer
	 * given this input
	 * @param input the input to use
	 * @return
	 */
	public Matrix computeOutputActivation(Matrix input) {
		this.input = input;
		return computeActivationOutput();
	}

	/**
	 * Sample this hidden layer given the input
	 * and initialize this layer with the given input
	 * @param input the input to sample
	 * @return the activation for this layer
	 * given the input
	 */
	public Matrix sampleHiddenGivenVisible_Data(Matrix input) {
		this.input = input;
				
		Matrix ret = MatrixUtils.genBinomialDistribution(computeActivationOutput(), 1, this.rndNumGenerator);
		
		return ret;
	}

	/**
	 * Sample this hidden layer given the last input.
	 * @return the activation for this layer given 
	 * the previous input
	 */
	public Matrix sampleHiddenGivenLastVisible() {
		Matrix output = computeActivationOutput();
		//reset the seed to ensure consistent generation of data
		//DoubleMatrix ret = MatrixUtil.binomial(output, 1, rng);
		Matrix ret = MatrixUtils.genBinomialDistribution(output, 1, this.rndNumGenerator);

		return ret;
	}	
	
	
	
	
	
}
