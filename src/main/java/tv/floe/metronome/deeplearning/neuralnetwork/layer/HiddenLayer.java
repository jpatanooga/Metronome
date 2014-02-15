package tv.floe.metronome.deeplearning.neuralnetwork.layer;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;


import tv.floe.metronome.deeplearning.neuralnetwork.activation.ActivationFunction;
import tv.floe.metronome.deeplearning.neuralnetwork.activation.Sigmoid;
import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseMultiLayerNeuralNetworkVectorized;
import tv.floe.metronome.math.MatrixUtils;

/**
 * Hidden Layer
 * 
 * TODO: explain the role of a hidden layer in a Deep Belief Network
 * - compared to a RBM, or a traditional hidden layer
 * 
 * 
 * @author josh
 *
 */
public class HiddenLayer implements Serializable {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 87885295280050784L;
	
	public int neuronCountPreviousLayer = 0;
	public int neuronCount = 0;
	
	public Matrix connectionWeights;
	public Matrix biasTerms;
	public RandomGenerator rndNumGenerator;
	public Matrix input;
	public ActivationFunction activationFunction = new Sigmoid();
	
	private HiddenLayer() {}
	
	
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
		
		if ( null == rndGen ) {
			this.rndNumGenerator = new MersenneTwister(1234);
		} else { 
			this.rndNumGenerator = rndGen;
		}
		

		NormalDistribution u = new NormalDistribution( this.rndNumGenerator, 0, .01, NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY );

		
		// init the connection weights
		this.connectionWeights = new DenseMatrix( this.neuronCountPreviousLayer, this.neuronCount );
		this.connectionWeights.assign(0.0);

		for (int r = 0; r < this.connectionWeights.numRows(); r++) {
			
			for(int c = 0; c < this.connectionWeights.numCols(); c++) { 
			
				this.connectionWeights.setQuick( r, c, u.sample() );
			
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
	
	
	
	/**
	 * Trigger an activation with the last specified input
	 * @return the activation of the last specified input
	 */
	public Matrix computeActivationOutput() {
		
/*		System.out println("computeActivationOutput()");
		System.out.println("input: rows: " + this.input.numRows() + ", cols: " + this.input.numCols() );
		System.out.println("connectionWeights: rows: " + this.connectionWeights.numRows() + ", cols: " + this.connectionWeights.numCols() );
	*/	
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
	
	@Override
	public HiddenLayer clone() {
		HiddenLayer layer = new HiddenLayer();
		layer.biasTerms = biasTerms.clone();
		layer.connectionWeights = connectionWeights.clone();
		layer.input = input.clone();
		layer.activationFunction = activationFunction;
		layer.neuronCount = neuronCount;
		layer.neuronCountPreviousLayer = neuronCountPreviousLayer;
		layer.rndNumGenerator = rndNumGenerator;
		return layer;
	}
	
	
	/**
	 * Serializes this to the output stream.
	 * @param os the output stream to write to
	 */
	public void write(OutputStream os) {
		try {

			ObjectOutputStream oos = new ObjectOutputStream(os);
		    DataOutput d = new DataOutputStream(os);
			
			d.writeInt( this.neuronCountPreviousLayer );
			d.writeInt( this.neuronCount );
			
			MatrixWritable.writeMatrix(d, this.connectionWeights );
			MatrixWritable.writeMatrix(d, this.biasTerms );
			
			oos.writeObject( this.rndNumGenerator );
			MatrixWritable.writeMatrix(d, this.input );
		    

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

			ObjectInputStream ois = new ObjectInputStream(is);
			DataInput di = new DataInputStream(is);
			
			this.neuronCountPreviousLayer = di.readInt();
			this.neuronCount = di.readInt();
			
			this.connectionWeights = MatrixWritable.readMatrix( di );
			this.biasTerms = MatrixWritable.readMatrix( di );
			
			this.rndNumGenerator = (RandomGenerator) ois.readObject();
			
			this.input = MatrixWritable.readMatrix( di );
			
			//BaseMultiLayerNeuralNetworkVectorized loaded = (BaseMultiLayerNeuralNetworkVectorized) ois.readObject();
			//update(loaded);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}
	
	
	
	
	
}
