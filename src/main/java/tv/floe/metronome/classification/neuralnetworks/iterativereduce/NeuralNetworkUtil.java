package tv.floe.metronome.classification.neuralnetworks.iterativereduce;

import java.util.ArrayList;



import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.Connection;
import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork.NetworkType;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;

public class NeuralNetworkUtil {
	
	private ArrayList<NeuralNetwork> worker_networks = new ArrayList<NeuralNetwork>();

	public void ClearNetworkBuffer() {
		
		this.worker_networks = new ArrayList<NeuralNetwork>();
		
	}
	
	public void AccumulateWorkerNetwork(NeuralNetwork worker_nn) {
		
		this.worker_networks.add( worker_nn );
		
	}
	
	public NeuralNetwork AverageNetworkWeights() throws Exception {
		
		int[] neuron_counts = new int[this.worker_networks.get(0).getLayersCount()];
		
		// probably not necesary w new clone conf setup
		for ( int x = 0; x < this.worker_networks.get(0).getLayersCount(); x++ ) {

			// get the counts of neurons for each layer on the first nn
			neuron_counts[x] = this.worker_networks.get(0).getLayerByIndex(x).getNeuronsCount();
			
		}
		
		// make sure all nn's have the same structure
		// get the neuron count for each layer, compare it to the collected first nn's number
		for (int x = 0; x < this.worker_networks.size(); x++ ) {
			
			if ( this.worker_networks.get(0).getLayerByIndex(x).getNeuronsCount() != neuron_counts[x] ) {
				throw new Exception("Invalid Neuron Counts For Network During Averaging Run");
			}
			
			
		}
		
		// TODO: we need some sort of factory setup to rebuild a network based on
		// the config object
		// create accumulation network to return
		NeuralNetwork averaged_network = this.buildAveragingNetworkFromConf(this.worker_networks.get(0).getConfig());
		// mock up the layer count / neuron counts
		
		
		for (int x = 0; x < this.worker_networks.size(); x++ ) {
			

			
		}
		
		
		return averaged_network;
	}
	
	private void accumulateNetwork(NeuralNetwork worker_network, NeuralNetwork summing_network) {
		
		// for each layer starting after the input layer
		for ( int x = 1; x < worker_network.getLayersCount(); x++ ) {
			
			this.accumulateLayer(worker_network.getLayerByIndex(x), summing_network.getLayerByIndex(x));
			
		}
		
	}
	
	private void accumulateLayer(Layer worker_layer, Layer summing_layer) {
		
		// for each neuron
        //for (Neuron neuron : worker_layer.getNeurons()) {
		for ( int x = 0; x < worker_layer.getNeuronsCount(); x++ ) {
        	
            
        }
		
		
		
	}
	
	private void accumulateNeuronConnections(Neuron workerNeuron, Neuron summingNeuron) {
		
        //for (Connection connection : worker_layer.getInConnections()) {
		for ( int x = 0; x < workerNeuron.inConnections.size(); x++ ) {
        	
            //connection.getWeight().setValue( rnd.nextDouble() );
			summingNeuron.getInConnections().get(x).setWeight( workerNeuron.getInConnections().get(x).getWeight() );
            
        }
		
	}
	
	public int getNetworkBufferCount() {
		
		return this.worker_networks.size();
		
	}
	
	/**
	 * So this is a poor substitute for a more formal factory setup
	 * 
	 * @param c
	 * @return
	 * @throws Exception 
	 */
	public NeuralNetwork buildAveragingNetworkFromConf(Config c) throws Exception {
		
		if (null != c.getConfValue("networkType")) {
			
			if (c.getConfValue("networkType").equals(NetworkType.MULTI_LAYER_PERCEPTRON)) {
				
				MultiLayerPerceptronNetwork mlp_network = new MultiLayerPerceptronNetwork();
								
				mlp_network.buildFromConf(c);		
				
				return mlp_network;
				
			} else {
				
				// other type of unsupported network -- throw exception and junk
				throw new Exception("Currently unsupported network type");
				
			}
			
		}
		
		return null;
		
	}
}
