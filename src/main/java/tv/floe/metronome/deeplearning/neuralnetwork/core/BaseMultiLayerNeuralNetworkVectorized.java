package tv.floe.metronome.deeplearning.neuralnetwork.core;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;
import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.nn.matrix.jblas.CDBN;
import com.ccc.deeplearning.nn.matrix.jblas.CRBM;
import com.ccc.deeplearning.nn.matrix.jblas.NeuralNetwork;
import com.ccc.deeplearning.nn.matrix.jblas.RBM;

import tv.floe.metronome.deeplearning.neuralnetwork.layer.HiddenLayer;
import tv.floe.metronome.deeplearning.neuralnetwork.layer.OutputLayer;

public abstract class BaseMultiLayerNeuralNetworkVectorized {

	public int inputNeuronCount;
	public int outputNeuronCount;
	
	//the hidden layer sizes at each layer
	public int[] hiddenLayerSizes;
	public int numberLayers;
	
	//the hidden layers
	public HiddenLayer[] hiddenLayers;	
	public OutputLayer outputLayer;
	
	// DA / RBM Layers
	public NeuralNetworkVectorized[] preTrainingLayers;
	
	public RandomGenerator randomGenerator;
	
	// the input data ---- how is this going to be handled?
	// how was it handled with the OOP-MLPN version?
	Matrix input = null;
	
	public double learningRateUpdate = 0.95;
	
	/**
	 * CTOR
	 * 
	 */
	public BaseMultiLayerNeuralNetworkVectorized() {
		
	}
	
		

	/**
	 * Base class for initializing the layers based on the input.
	 * This is meant for capturing numbers such as input columns or other things.
	 * 
	 * This method sets up two types of layers:
	 * - normal ML-NN layers
	 * - RBM / DA layers
	 * 
	 * @param input the input matrix for training
	 */
	protected void initializeLayers(Matrix input) {
		Matrix layer_input = input;
		int input_size;

		// construct multi-layer
		for (int i = 0; i < this.numberLayers; i++) {
			
			if (i == 0) { 
				
				//input_size = this.nIns;
				input_size = this.inputNeuronCount;

				// construct sigmoid_layer
				//this.sigmoidLayers[i] = new HiddenLayer(input_size, this.hiddenLayerSizes[i], null, null, rng,layer_input);
				this.hiddenLayers[ i ] = new HiddenLayer(input_size, this.hiddenLayerSizes[i], this.randomGenerator );
				this.hiddenLayers[ i ].setInput( layer_input );

			} else {
				
				input_size = this.hiddenLayerSizes[ i - 1 ];
				//layer_input = sigmoidLayers[i - 1].sample_h_given_v();
				layer_input = this.hiddenLayers[i - 1].sampleHiddenGivenLastVisible();
				// construct sigmoid_layer
				//this.sigmoidLayers[i] = new HiddenLayer(input_size, this.hiddenLayerSizes[i], null, null, rng,layer_input);
				this.hiddenLayers[ i ] = new HiddenLayer(input_size, this.hiddenLayerSizes[i], this.randomGenerator);
				this.hiddenLayers[ i ].setInput( layer_input );
				
			}


			// construct dA_layer
			this.preTrainingLayers[ i ] = createPreTrainingLayer( layer_input,input_size, this.hiddenLayerSizes[i], this.hiddenLayers[i].connectionWeights, this.hiddenLayers[i].biasTerms, null, this.randomGenerator, i );
		}

		this.outputLayer = new OutputLayer(layer_input, this.hiddenLayerSizes[this.numberLayers-1], this.outputNeuronCount );

	}
	
	/**
	 * Creates a layer depending on the index.
	 * The main reason this matters is for continuous variations such as the {@link CDBN}
	 * where the first layer needs to be an {@link CRBM} for continuous inputs
	 * 
	 */
	public abstract NeuralNetworkVectorized createPreTrainingLayer(Matrix input, int nVisible, int nHidden, Matrix weights, Matrix hbias, Matrix vBias, RandomGenerator rng, int index);
	


	public void finetune(double lr, int epochs) {
		finetune(this.labels,lr,epochs);

	}

	/**
	 * Run SGD based on the given labels
	 * @param labels the labels to use
	 * @param lr the learning rate during training
	 * @param epochs the number of times to iterate
	 */
	public void finetune(Matrix labels,double lr, int epochs) {
		optimizer = new MultiLayerNetworkOptimizer(this,lr);
		optimizer.optimize(labels, lr,epochs);
	}



	/**
	 * Label the probabilities of the input
	 * @param x the input to label
	 * @return a vector of probabilities
	 * given each label.
	 * 
	 * This is typically of the form:
	 * [0.5, 0.5] or some other probability distribution summing to one
	 */
	public Matrix predict(Matrix x) {
		Matrix input = x;
		for(int i = 0; i < nLayers; i++) {
			HiddenLayer layer = sigmoidLayers[i];
			input = layer.outputMatrix(input);
		}
		return logLayer.predict(input);
	}


	/**
	 * Serializes this to the output stream.
	 * @param os the output stream to write to
	 */
	public void write(OutputStream os) {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(os);
			oos.writeObject(this);

		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}


	/**
	 * Negative log likelihood of the model
	 * @return the negative log likelihood of the model
	 */
	public double negativeLogLikelihood() {
		return logLayer.negativeLogLikelihood();
	}
	
	/**
	 * Train the network running some unsupervised 
	 * pretraining followed by SGD/finetune
	 * @param input the input to train on
	 * @param labels the labels for the training examples(a matrix of the following format:
	 * [0,1,0] where 0 represents the labels its not and 1 represents labels for the positive outcomes 
	 * @param otherParams the other parameters for child classes (algorithm specific parameters such as corruption level for SDA)
	 */
	public abstract void trainNetwork(Matrix input,Matrix labels,Object[] otherParams);

	/**
	 * Creates a layer depending on the index.
	 * The main reason this matters is for continuous variations such as the {@link CDBN}
	 * where the first layer needs to be an {@link CRBM} for continuous inputs
	 * @param input the input to the layer
	 * @param nVisible the number of visible inputs
	 * @param nHidden the number of hidden units
	 * @param W the weight vector
	 * @param hbias the hidden bias
	 * @param vBias the visible bias
	 * @param rng the rng to use (THiS IS IMPORTANT; YOU DO NOT WANT TO HAVE A MIS REFERENCED RNG OTHERWISE NUMBERS WILL BE MEANINGLESS)
	 * @param index the index of the layer
	 * @return a neural network layer such as {@link RBM} 
	 */
//	public abstract NeuralNetwork createLayer(Matrix input,int nVisible,int nHidden, Matrix W,Matrix hbias,Matrix vBias,RandomGenerator rng,int index);


//	public abstract NeuralNetwork[] createNetworkLayers(int numLayers);


	
}
