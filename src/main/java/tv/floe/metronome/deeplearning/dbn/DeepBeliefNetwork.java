package tv.floe.metronome.deeplearning.dbn;

import java.util.ArrayList;

import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;
import tv.floe.metronome.deeplearning.rbm.RestrictedBoltzmannMachine;

/**
 * Base draft of a Deep Belief Network based on RBMs
 * (based on concepts by Hinton)
 * 
 *  1. Setup as a normal MLPN 
 *  - but it also has a set of RBM layers that matches the number of hidden layers
 *  
 *  2. as each RBM is trained
 *  - its weights and bias are transferred into the weights/bias of the MLPN
 * 
 * @author josh
 *
 */
public class DeepBeliefNetwork {
	
	private MultiLayerPerceptronNetwork mlpn = null;
	private ArrayList<RestrictedBoltzmannMachine> rbmLayers = null;

	public DeepBeliefNetwork() {
		
		
	}
	
	public void train() {
		
	}
	
	public void preTrain() {
		
		
	}
	
	
	
}
