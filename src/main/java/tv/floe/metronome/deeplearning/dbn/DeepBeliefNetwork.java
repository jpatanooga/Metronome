package tv.floe.metronome.deeplearning.dbn;


import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.commons.math3.random.RandomGenerator;

import tv.floe.metronome.deeplearning.math.transforms.MatrixTransform;
import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseMultiLayerNeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.core.LogisticRegression;
import tv.floe.metronome.deeplearning.neuralnetwork.core.NeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.layer.HiddenLayer;
import tv.floe.metronome.deeplearning.neuralnetwork.optimize.MultiLayerNetworkOptimizer;
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
	
	//private RandomGenerator randomGen = new MersenneTwister(1234);
	
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
		
		
		if (this.inputTrainingData == null || this.hiddenLayers == null || this.hiddenLayers[0] == null || this.logisticRegressionLayer == null) {
			this.inputTrainingData = trainingRecords;
			initializeLayers(trainingRecords);
		} else {
			System.out.println( "PreTrain > Setting Input..." );
			this.inputTrainingData = trainingRecords;
		}
		
		Matrix layerInput = null;
		
		for (int i = 0; i < this.numberLayers; i++) {
			
			System.out.println("PreTrain > Layer " + i );
			
			if (i == 0) {
				
				layerInput = this.inputTrainingData;
			
			} else { 
			
				//based on the previous layer input, let's see what representation it learned
				layerInput = hiddenLayers[ i - 1 ].sampleHiddenGivenVisible_Data(layerInput);
			
			}
			
			this.preTrainingLayers[ i ].trainTillConvergence( layerInput, learningRate, new Object[]{ k, learningRate, epochs } );

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
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine( input, nVisible, nHidden, weights, hbias, vBias, rng ); 
		
		rbm.useRegularization = this.useRegularization;
		rbm.setSparsity( this.getSparsity() );
		rbm.setMomentum( this.getMomentum() );
		
		return rbm;
		
	}

	@Override
	public NeuralNetworkVectorized[] createNetworkLayers(int numLayers) {
	
		return new RestrictedBoltzmannMachine[numLayers];	
		
	}
	
	/**
	 * Serializes this to the output stream.
	 * 
	 * Current Thoughts:
	 * - cant save entire training dataset in an IterativeReduce scenario,
	 * 		- would copy entire dataset around the network!
	 * 
	 * TODO: 
	 * - this about this critically: what is the minimum amount of info we can 
	 * 	 get away w sending for parameter-averaging
	 * 
	 * @param os the output stream to write to
	 */
	public void write(OutputStream os) {
		try {

		    DataOutput d = new DataOutputStream(os);
		    ObjectOutputStream oos = new ObjectOutputStream(os);
		 
		    
		    d.writeInt( this.inputNeuronCount );
		    d.writeInt( this.outputNeuronCount );
		    
		    d.writeInt( this.numberLayers );
		    
		    for ( int x = 0; x < this.numberLayers; x++ ) {
		    	
		    	d.writeInt( this.hiddenLayerSizes[ x ] );
		    	
		    }

		    // write in hidden layers
		    for ( int x = 0; x < this.numberLayers; x++ ) {

		    	this.hiddenLayers[ x ].write( os );
		    	
		    }
		    
		    this.logisticRegressionLayer.write( os );
		    
			// DA / RBM Layers
		    for ( int x = 0; x < this.numberLayers; x++ ) {

		    	((RestrictedBoltzmannMachine)this.preTrainingLayers[ x ]).write( os );
		    	
		    }
		    
		    oos.writeObject( this.randomGenerator );
		    oos.writeObject( this.distribution );

		    
//		    MatrixWritable.writeMatrix(d, this.inputTrainingData );
//		    MatrixWritable.writeMatrix(d, this.outputTrainingLabels );

			d.writeDouble( this.learningRateUpdate );
			d.writeBoolean( this.useRegularization );
			d.writeDouble( this.l2 );
			
			d.writeDouble( this.getMomentum() );
			d.writeDouble( this.getSparsity() );
			

			// dont serde optimizer
			
			// TODO: weight transforms

			//  Map<Integer,MatrixTransform> weightTransforms
		    

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
			
		    ObjectInputStream ois = new ObjectInputStream(is);
		    
		    this.inputNeuronCount = di.readInt();
		    this.outputNeuronCount = di.readInt();
		    this.numberLayers = di.readInt();
		    
		    this.hiddenLayerSizes = new int[this.numberLayers];
		    for ( int x = 0; x < this.numberLayers; x++ ) {
		    	
		    	this.hiddenLayerSizes[ x ] = di.readInt();
		    	
		    }
		    
		    this.hiddenLayers = new HiddenLayer[ this.numberLayers ];
		    // write in hidden layers
		    for ( int x = 0; x < this.numberLayers; x++ ) {

		    	this.hiddenLayers[ x ] = new HiddenLayer( 1, 1, null); 
		    	this.hiddenLayers[ x ].load( is );
		    	
		    	
		    }
		    
		    
		    // this.logisticRegressionLayer = new LogisticRegression(layer_input, this.hiddenLayerSizes[this.numberLayers-1], this.outputNeuronCount );
		    this.logisticRegressionLayer = new LogisticRegression();
		    this.logisticRegressionLayer.load(is);
		    
		    this.preTrainingLayers = new RestrictedBoltzmannMachine[ this.numberLayers ];
		    for ( int x = 0; x < this.numberLayers; x++ ) {

		    	RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(1, 1, null);
		    	rbm.load(is);
		    	this.preTrainingLayers[ x ] = rbm;
		    	
		    }
		    
		    this.randomGenerator = (RandomGenerator) ois.readObject();
		    this.distribution = (RealDistribution) ois.readObject();

//		    this.inputTrainingData = MatrixWritable.readMatrix( di );	
//		    this.outputTrainingLabels = MatrixWritable.readMatrix( di );

		    this.learningRateUpdate = di.readDouble();
		    this.useRegularization = di.readBoolean();
		    this.l2 = di.readDouble();
		    
		    this.setMomentum( di.readDouble() );
		    this.setSparsity( di.readDouble() );
		    
		    
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

	    // write in hidden layers
	    for ( int x = 0; x < this.numberLayers; x++ ) {

	    	this.hiddenLayers[ x ].write( os );
	    	
	    }
		
	    this.logisticRegressionLayer.write( os );
	    
		// DA / RBM Layers
	    for ( int x = 0; x < this.numberLayers; x++ ) {

	    	((RestrictedBoltzmannMachine)this.preTrainingLayers[ x ]).write( os );
	    	
	    }
	    		    

	}		
	
	/**
	 * Load parameter values from the byte stream 
	 * 
	 */
	public void loadParameterValues(InputStream is) {
		try {

			DataInput di = new DataInputStream(is);
			
		    this.hiddenLayers = new HiddenLayer[ this.numberLayers ];
		    // write in hidden layers
		    for ( int x = 0; x < this.numberLayers; x++ ) {

		    	this.hiddenLayers[ x ] = new HiddenLayer( 1, 1, null); 
		    	this.hiddenLayers[ x ].load( is );
		    	
		    	
		    }
		    
		    
		    // this.logisticRegressionLayer = new LogisticRegression(layer_input, this.hiddenLayerSizes[this.numberLayers-1], this.outputNeuronCount );
		    this.logisticRegressionLayer = new LogisticRegression();
		    this.logisticRegressionLayer.load(is);
		    
		    this.preTrainingLayers = new RestrictedBoltzmannMachine[ this.numberLayers ];
		    for ( int x = 0; x < this.numberLayers; x++ ) {

		    	this.preTrainingLayers[ x ] = new RestrictedBoltzmannMachine(1, 1, null);
		    	((RestrictedBoltzmannMachine)this.preTrainingLayers[ x ]).load(is);
		    	//this.preTrainingLayers[ x ] = rbm;
		    	
		    }
		    				
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}		

	
	
	
}
