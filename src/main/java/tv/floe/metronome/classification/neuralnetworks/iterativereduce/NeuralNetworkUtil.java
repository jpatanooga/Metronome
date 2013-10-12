package tv.floe.metronome.classification.neuralnetworks.iterativereduce;

import java.util.ArrayList;

import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;

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
		
		// create accumulation network to return
		NeuralNetwork averaged_network = new NeuralNetwork();
		
		for (int x = 0; x < this.worker_networks.size(); x++ ) {
			
			//this.worker_networks.get(x)
			
			
		}
		
		
		return averaged_network;
	}
	
	public int getNetworkBufferCount() {
		
		return this.worker_networks.size();
		
	}
}
