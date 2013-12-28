package tv.floe.metronome.deeplearning.dbn;

import java.util.ArrayList;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;
import tv.floe.metronome.deeplearning.rbm.RestrictedBoltzmannMachine;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

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
 * @author josh
 * 
 * TODO
 * 
 * -	decide on a strategy for working with input samples
 * 		-	as we going to look at all the samples in memory at once for the MLPN?
 * 		-	how do we balance out the implementation styles?
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
 * TODO:
 * - need to figure out how the greedy-per-layer SGD pass works in this case
 *
 *
 */
public class DeepBeliefNetwork {
	
	private MultiLayerPerceptronNetwork mlpn = null;
	private ArrayList<RestrictedBoltzmannMachine> rbmLayers = null;

	private int preTrainEpochCount = 1000;
	private double learningRate = 0.1d;
	
	private RandomGenerator randomGen = new MersenneTwister(1234);
	
	public DeepBeliefNetwork() {
		
		
	}
	
	/**
	 * create RBM layers based on the layers in the MLPN
	 * 
	 * @param nn
	 */
	public DeepBeliefNetwork(MultiLayerPerceptronNetwork nn) {
		
		
	}
	
	/**
	 * train against a single record coming in w its expected output
	 * 
	 * - mainly intended to be used in a streaming/online fashion, 
	 * 		such as with a file reader in a hadoop env
	 * 
	 * 
	 */
	public void trainOnSingleRecord(Vector actualOutVals, Vector trainingInstance) {
		
		// 1. pre-train
		
		
		// 2. fine tune
		
	}

	/**
	 * train against the entire dataset cached into a matrix of rows
	 * - also includes the matrix of the outputs for each input record
	 * 
	 * 
	 */
	public void trainOnEntireDataset(Matrix trainingRecordOutputs, Matrix trainingRecords) {
		
		// 1. pre-train
		
		
		// 2. fine tune
		
	}
	
	
	/**
	 * This is where we work through each RBM layer, learning an unsupervised 
	 * representation of the data
	 * 
	 * TODO: make sure layers match and the input layer isnt messing up counts
	 * 
	 * TODO: 
	 * 
	 */
	public void preTrain(Matrix trainingRecords) {
		
		Matrix layerTrainingInput = trainingRecords;
		
		// for each of the RBM layers..
		for ( int x = 0; x < this.rbmLayers.size(); x++ ) {
			
						
			RestrictedBoltzmannMachine rbmLayer = this.rbmLayers.get(x);
			
			// TODO: GET reference to the layer in the MLPN that matches the rbmLayer
			// we'll update its weights based on what we learn from the rbmLayer
			
			Layer mlpnLayer = this.mlpn.getLayerByIndex(x);
			
			for ( int epoch = 0; epoch < this.preTrainEpochCount; epoch++ ) {
				
				// 1. run CDk on the current rbmLayer
				rbmLayer.contrastiveDivergence( 1, layerTrainingInput );
				
				// 2. now update the current hidden layer for the MLPN with the 
				// weights and bias terms from the rbmLayer
				
				Vector weights = rbmLayer.connectionWeights.viewRow(0);
				
				mlpnLayer.loadConnectingWeights( weights );
				
				// TODO: figure out the bias issue here w MLPN
				
				
				
			}
			
			// now swap the training input
			// layerInput = sigmoidLayers[i-1].sample_h_given_v(layerInput);
			
			// so based off the training of the previous layer, we want to see what the output
			// of this previous layer looks like via "sample_h_given_v( layerTrainingInput )"
			
			// basically we're saying "based on distributions of the data (and input from last layer) give us a 
			// representation to use as input for the next layer"
			
			// TODO: thought: cna we not simply sample directly from the RBM itself?
			// - what advantage is there loading into the hidden layer and then sampling from there?
			
			layerTrainingInput = this.sampleHiddenGivenVisible(layerTrainingInput, mlpnLayer);
			
		}
		
		
	}
	
	/**
	 * TODO: look at this more closely
	 * 
	 * 
	 * @return
	 */
	public Matrix outputMatrix( Matrix input ) {
//		Matrix mult = input.mmul(W);
//		mult = mult.addRowVector(b);
//		return MatrixUtil.sigmoid(mult);
		return null;
	}
	
	
	/**
	 * This function infers state of visible units given hidden units
	 * 
	 *  
	 * @param input
	 * @param mlpnLayer
	 * @return
	 */
	public Matrix sampleHiddenGivenVisible(Matrix visible, Layer mlpnLayer) {
		
		
/*		
		Matrix visibleProb = this.generateProbabilitiesForVisibleStatesBasedOnHiddenStates(hidden);

		Matrix visibleBinomialSample = MatrixUtils.genBinomialDistribution(visibleProb, 1, this.randNumGenerator);

		return new Pair<Matrix, Matrix>(visibleProb, visibleBinomialSample);
	*/	
		
		//reset the seed to ensure consistent generation of data
		//Matrix ret = MatrixUtils.binomial(outputMatrix(input), 1, this.randomGen);
		Matrix ret = MatrixUtils.genBinomialDistribution( outputMatrix(visible), 1, this.randomGen);
		//return ret;
		return ret;
	}

	
	
	
}
