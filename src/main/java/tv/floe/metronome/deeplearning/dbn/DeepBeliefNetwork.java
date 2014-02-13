package tv.floe.metronome.deeplearning.dbn;


import org.apache.commons.math3.random.MersenneTwister;
import org.apache.mahout.math.Matrix;
import org.apache.commons.math3.random.RandomGenerator;

import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseMultiLayerNeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.core.NeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.layer.HiddenLayer;
import tv.floe.metronome.deeplearning.rbm.RestrictedBoltzmannMachine;

/**
 * Base draft of a Deep Belief Network based on RBMs
 * (based on concepts by Hinton)
 * 
 * Literature Review and Notes
 * 
 * 1. http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DBNPseudoCode
 * 
 * 
 *  1. Setup as a normal MLPN 
 *  - but it also has a set of RBM layers that matches the number of hidden layers
 *  
 *  2. as each RBM is trained
 *  - its weights and bias are transferred into the weights/bias of the MLPN
 * 
 * Deep Belief Network. This is a MultiLayer Perceptron Model
 * using Restricted Boltzmann Machines.
 *  See Hinton's practical guide to RBMs for great examples on
 *  how to train and tune parameters.
 * 
 * 
 * @author josh
 * 
 *
 *
 * TODO:
 * - thoughts: are we going to train each layer separately as a yarn job?
 * 		- if so, how do we coordinate that?
 * 		- who tracks layer stuff, and how?
 * 
 * - IR: as long as all the workers advance their layer positing in sync, we should be good
 * 		to make this as one continuous job
 * 		-	need a way to save layers in progress to view in viewer
 *
 *
 *
 */
public class DeepBeliefNetwork extends BaseMultiLayerNeuralNetworkVectorized {
	
	private RandomGenerator randomGen = new MersenneTwister(1234);
	
	// default CTOR
	public DeepBeliefNetwork() {
		
		
	}
	
	public DeepBeliefNetwork(int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers,
			RandomGenerator rng, Matrix input, Matrix labels) {
		super(n_ins, hidden_layer_sizes, n_outs, n_layers, rng, input,labels);
	}



	public DeepBeliefNetwork(int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers,
			RandomGenerator rng) {
		super(n_ins, hidden_layer_sizes, n_outs, n_layers, rng);
	}


	
	
	/**
	 * This is where we work through each RBM layer, learning an unsupervised 
	 * representation of the data
	 * 
	 * This unsupervised learning method runs
	 * contrastive divergence on each RBM layer in the network.
	 * @param trainingRecords the input to train on
	 * @param k the k to use for running the RBM contrastive divergence.
	 * 
	 * The typical tip is that the higher k is the closer to the model
	 * you will be approximating due to more sampling. K = 1
	 * usually gives very good results and is the default in quite a few situations.
	 * 
	 * The training input to each successive layer is the activations of the hidden layer of the previous pretraining layer's RBM Hidden Neurons
	 * 
	 * 
	 */
	public void preTrain(Matrix trainingRecords,int k,double learningRate,int epochs) {
		
		if (this.inputTrainingData == null) {
			this.inputTrainingData = trainingRecords;
			initializeLayers(trainingRecords);
		}
		
		Matrix layerInput = null;
		
		for (int i = 0; i < this.numberLayers; i++) {
			
			System.out.println("PreTrain > Layer " + i );
			
			if (i == 0) {
				layerInput = this.inputTrainingData;
			} else { 
				layerInput = hiddenLayers[ i - 1 ].sampleHiddenGivenVisible_Data(layerInput);
			}

			this.preTrainingLayers[i].trainTillConvergence(layerInput, learningRate, new Object[]{k});

		}
		
	}
	
	/**
	 * Trains the Deep Belief Network
	 * 
	 * Step 1: Pretrain the RBM layers with Contrastive Divergence
	 * 
	 * Step 2: Finetune the NN layers with gentle backpropagation
	 * 
	 */
	@Override
	public void trainNetwork(Matrix input, Matrix labels, Object[] otherParams) {

		int k = (Integer) otherParams[0];
		double learningRate = (Double) otherParams[1];
		int epochs = (Integer) otherParams[2];
		
		System.out.println( "Training Network... " );

		preTrain(input, k, learningRate, epochs);
		
		if (otherParams.length < 3) {
			
			finetune(labels, learningRate, epochs);
			
		} else {
			
			double finetuneLr = otherParams.length > 3 ? (Double) otherParams[3] : learningRate;
			int finetuneEpochs = otherParams.length > 4 ? (Integer) otherParams[4] : epochs;
			finetune( labels, finetuneLr, finetuneEpochs);
			
		}

	}	
	
	
	

	@Override
	public NeuralNetworkVectorized createPreTrainingLayer(Matrix input,
			int nVisible, int nHidden, Matrix weights, Matrix hbias,
			Matrix vBias, RandomGenerator rng, int index) {
		
		return new RestrictedBoltzmannMachine( input, nVisible, nHidden, weights, hbias, vBias, rng );
		
	}

	@Override
	public NeuralNetworkVectorized[] createNetworkLayers(int numLayers) {
	
		return new RestrictedBoltzmannMachine[numLayers];	
		
	}

	
	
	
}
