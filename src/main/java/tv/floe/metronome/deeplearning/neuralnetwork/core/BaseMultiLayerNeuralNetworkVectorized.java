package tv.floe.metronome.deeplearning.neuralnetwork.core;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.neuralnetwork.activation.ActivationFunction;
import tv.floe.metronome.deeplearning.neuralnetwork.layer.HiddenLayer;
import tv.floe.metronome.deeplearning.neuralnetwork.optimize.MultiLayerNetworkOptimizer;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

public abstract class BaseMultiLayerNeuralNetworkVectorized {

	public int inputNeuronCount;
	public int outputNeuronCount;
	
	//the hidden layer sizes at each layer
	public int[] hiddenLayerSizes;
	public int numberLayers;
	
	//the hidden layers
	public HiddenLayer[] hiddenLayers;	
	
	// TODO: do we rename this to LogisticRegressionOutputLayer ?
	public LogisticRegression logisticRegressionLayer;
	
	// DA / RBM Layers
	public NeuralNetworkVectorized[] preTrainingLayers;
	
	public RandomGenerator randomGenerator;
	
	// the input data ---- how is this going to be handled?
	// how was it handled with the OOP-MLPN version?
	public Matrix inputTrainingData = null;
	public Matrix outputTrainingLabels = null;
	
	public double learningRateUpdate = 0.95;
	public boolean useRegularization = true;
	public double l2 = 0.01;
	
	
	public MultiLayerNetworkOptimizer optimizer;
	
	/**
	 * CTOR
	 * 
	 */
	public BaseMultiLayerNeuralNetworkVectorized() {
		
	}
	
	public BaseMultiLayerNeuralNetworkVectorized(int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers, RandomGenerator rng) {
		
		this(n_ins,hidden_layer_sizes,n_outs,n_layers,rng,null,null);
		
	}


	public BaseMultiLayerNeuralNetworkVectorized(int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers, RandomGenerator rng, Matrix input, Matrix labels) {
		
		this.inputNeuronCount = n_ins;
		this.hiddenLayerSizes = hidden_layer_sizes;
		this.inputTrainingData = input;
		this.outputTrainingLabels = labels;

		if(hidden_layer_sizes.length != n_layers) {
			throw new IllegalArgumentException("The number of hidden layer sizes must be equivalent to the nLayers argument which is a value of " + n_layers);
		}

		this.outputNeuronCount = n_outs;
		this.numberLayers = n_layers;

		this.hiddenLayers = new HiddenLayer[n_layers];
		this.preTrainingLayers = createNetworkLayers( this.numberLayers );

		if (rng == null) {   
			this.randomGenerator = new MersenneTwister(123);
		} else { 
			this.randomGenerator = rng;
		}


		if (input != null) { 
			initializeLayers(input);
		}


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

		System.out.println("initializeLayers --------");
		
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
			
			System.out.println("Created Hidden Layer: ConnWeight Matrix - rows: " + this.hiddenLayers[i].connectionWeights.numRows() + ", cols " + this.hiddenLayers[i].connectionWeights.numCols() );


			// construct DL appropriate class for pre training layer
			this.preTrainingLayers[ i ] = createPreTrainingLayer( layer_input,input_size, this.hiddenLayerSizes[i], this.hiddenLayers[i].connectionWeights, this.hiddenLayers[i].biasTerms, null, this.randomGenerator, i );
		}

		this.logisticRegressionLayer = new LogisticRegression(layer_input, this.hiddenLayerSizes[this.numberLayers-1], this.outputNeuronCount );

		System.out.println( "Finished layer init ------  " );
		
	}
	

	public List<Matrix> feedForward() {
		
		if (this.inputTrainingData == null) {
			throw new IllegalStateException("Unable to perform feed forward; no input found");
		}
		
		List<Matrix> activations = new ArrayList<Matrix>();
		Matrix input = this.inputTrainingData;
		activations.add(input);

		for (int i = 0; i < this.numberLayers; i++) {
			
			HiddenLayer layer = this.hiddenLayers[ i ];
			//layers[i].setInput(input);
			this.preTrainingLayers[i].setInput(input);
			input = layer.computeOutputActivation(input);
			activations.add(input);
			
		}

		activations.add( this.logisticRegressionLayer.predict(input) );
		return activations;
	}

	/**
	 * TODO: make sure our concept of an activation function can deliver functionality
	 * 
	 * @param activations
	 * @param deltaRet
	 */
	private void computeDeltas(List<Matrix> activations,List<Pair<Matrix,Matrix>> deltaRet) {
		
		Matrix[] gradients = new Matrix[ this.numberLayers + 2 ];
		Matrix[] deltas = new Matrix[ this.numberLayers + 2 ];
		ActivationFunction derivative = this.hiddenLayers[ 0 ].activationFunction;
		
		//- y - h
		Matrix delta = null;


		/*
		 * Precompute activations and z's (pre activation network outputs)
		 */
		List<Matrix> weights = new ArrayList<Matrix>();
/*		
		for (int j = 0; j < layers.length; j++) {
			weights.add(layers[j].getW());
		}
*/
		for (int j = 0; j < this.preTrainingLayers.length; j++) {
			
			weights.add( this.preTrainingLayers[j].getConnectionWeights() );
			
		}
		
		weights.add( this.logisticRegressionLayer.connectionWeights );

		List<Matrix> zs = new ArrayList<Matrix>();
		
		zs.add( this.inputTrainingData );
		
		for (int i = 0; i < this.preTrainingLayers.length; i++) {
			
			if (this.preTrainingLayers[i].getInput() == null && i == 0) {
				
				this.preTrainingLayers[i].setInput( this.inputTrainingData );
				
			} else if (this.preTrainingLayers[i].getInput() == null) {
				
				this.feedForward();
				
			}

			//zs.add(MatrixUtil.sigmoid(layers[i].getInput().mmul(weights.get(i)).addRowVector(layers[i].gethBias())));
			//zs.add(MatrixUtils.sigmoid( this.preTrainingLayers[ i ].getInput().times( weights.get( i ) ).addRowVector( this.preTrainingLayers[i].getHiddenBias() )));
			zs.add(MatrixUtils.sigmoid( MatrixUtils.addRowVector( this.preTrainingLayers[ i ].getInput().times( weights.get( i ) ),  this.preTrainingLayers[i].getHiddenBias().viewRow(0) )));
			
		}
		
		//zs.add(logLayer.input.mmul(logLayer.W).addRowVector(logLayer.b));
		//zs.add( this.logisticRegressionLayer.input.times( this.logisticRegressionLayer.connectionWeights ).addRowVector( this.logisticRegressionLayer.biasTerms ));
		zs.add( MatrixUtils.addRowVector( this.logisticRegressionLayer.input.times( this.logisticRegressionLayer.connectionWeights ), this.logisticRegressionLayer.biasTerms.viewRow(0) ) );

		//errors
		for (int i = this.numberLayers + 1; i >= 0; i--) {
			
			if (i >= this.numberLayers + 1) {
				
				Matrix z = zs.get(i);
				//- y - h
				//delta = labels.sub(activations.get(i)).neg();
				delta = MatrixUtils.neg( this.outputTrainingLabels.minus( activations.get( i ) ) );

				//(- y - h) .* f'(z^l) where l is the output layer
				Matrix initialDelta = delta.times( derivative.applyDerivative( z ) );
				deltas[ i ] = initialDelta;

			} else {
				
				delta = deltas[ i + 1 ];
				Matrix w = weights.get( i ).transpose();
				Matrix z = zs.get( i );
				Matrix a = activations.get( i + 1 );
				//W^t * error^l + 1

				Matrix error = delta.times( w );
				deltas[ i ] = error;

				error = error.times(derivative.applyDerivative(z));

				deltas[ i ] = error;
//				gradients[ i ] = a.transpose().times(error).transpose().div( this.inputTrainingData.numRows() );
				gradients[ i ] = a.transpose().times(error).transpose().divide( this.inputTrainingData.numRows() );
			}

		}


		for (int i = 0; i < gradients.length; i++) {
			
			deltaRet.add(new Pair<Matrix, Matrix>(gradients[i],deltas[i]));
			
		}
		
	}
	
	
	/**
	 * Backpropagation of errors for weights
	 * @param lr the learning rate to use
	 * @param epochs  the number of epochs to iterate (this is already called in finetune)
	 */
	public void backProp(double lr,int epochs) {
		
		for (int i = 0; i < epochs; i++) {
			
			List<Matrix> activations = feedForward();

			//precompute deltas
			List<Pair<Matrix,Matrix>> deltas = new ArrayList<Pair<Matrix, Matrix>>();
			computeDeltas(activations, deltas);


			for (int l = 0; l < this.numberLayers; l++) {
				
				Matrix add = deltas.get( l ).getFirst().divide( this.inputTrainingData.numRows() ).times( lr );
				
				add = add.divide( this.inputTrainingData.numRows() );
				
				if (useRegularization) {
					
					add = add.times( this.preTrainingLayers[ l ].getConnectionWeights().times( l2 ) );
					
				}


				this.preTrainingLayers[ l ].setConnectionWeights( this.preTrainingLayers[ l ].getConnectionWeights().minus( add.times( lr ) ) );
				
				this.hiddenLayers[ l ].connectionWeights = this.preTrainingLayers[l].getConnectionWeights();
				Matrix deltaColumnSums = MatrixUtils.columnSums( deltas.get( l + 1 ).getSecond() );
				
				// TODO: check this, needs to happen in place?
				deltaColumnSums = deltaColumnSums.divide( this.inputTrainingData.numRows() );

				// TODO: check this, needs to happen in place?
				//this.preTrainingLayers[ l ].getHiddenBias().subi( deltaColumnSums.times( lr ) );
				Matrix hbiasMinus = this.preTrainingLayers[ l ].getHiddenBias().minus( deltaColumnSums.times( lr ) );
				this.preTrainingLayers[ l ].sethBias(hbiasMinus);
				
				this.hiddenLayers[ l ].biasTerms = this.preTrainingLayers[l].getHiddenBias();
			}


			this.logisticRegressionLayer.connectionWeights = this.logisticRegressionLayer.connectionWeights.plus(deltas.get( this.numberLayers ).getFirst());


		}





	}	
	
	/**
	 * Creates a layer depending on the index.
	 * The main reason this matters is for continuous variations such as the {@link CDBN}
	 * where the first layer needs to be an {@link CRBM} for continuous inputs
	 * 
	 */
	public abstract NeuralNetworkVectorized createPreTrainingLayer(Matrix input, int nVisible, int nHidden, Matrix weights, Matrix hbias, Matrix vBias, RandomGenerator rng, int index);
	


	public void finetune(double learningRate, int epochs) {
		
		finetune( this.outputTrainingLabels, learningRate, epochs );

	}

	/**
	 * Run SGD based on the given output vectors
	 * 
	 * 
	 * @param labels the labels to use
	 * @param lr the learning rate during training
	 * @param epochs the number of times to iterate
	 */
	public void finetune(Matrix outputLabels, double learningRate, int epochs) {
		
		optimizer = new MultiLayerNetworkOptimizer(this,learningRate);
		optimizer.optimize( outputLabels, learningRate, epochs );
		
	}


	/**
	 * Label the probabilities of the input
	 * @param x the input to label
	 * @return a vector of probabilities
	 * given each label.
	 * 
	 * This is typically of the form:
	 * [0.5, 0.5] or some other probability distribution summing to one
	 * 
	 * 
	 */
	public Matrix predict(Matrix x) {
		
		Matrix input = x;
		
		for(int i = 0; i < this.numberLayers; i++) {
			HiddenLayer layer = this.hiddenLayers[i];
			input = layer.computeOutputActivation(input);
		}
		
		return this.logisticRegressionLayer.predict(input);
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
	 * @return the negative log likelihood of the model
	 */
	public double negativeLogLikelihood() {
		return this.logisticRegressionLayer.negativeLogLikelihood();
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


	public abstract NeuralNetworkVectorized[] createNetworkLayers(int numLayers);


	
}
