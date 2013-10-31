package tv.floe.metronome.classification.neuralnetworks.utils;

import org.apache.mahout.math.Vector;

import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;

public class Utils {
	

	public static void PrintNeuralNetwork(NeuralNetwork nn) {
		
		
		for ( int x = 1; x < nn.getLayersCount(); x++ ) {
			
			System.out.println("Layer " + x );
			
			for ( int n = 0; n < nn.getLayerByIndex(x).getNeuronsCount(); n++ ) {
				
				System.out.println("\tNeuron " + n);
				
				for ( int c = 0; c < nn.getLayerByIndex(x).getNeuronAt(n).inConnections.size(); c++) {
					
					System.out.println("\t\tConnection " + c + ", weight: " + nn.getLayerByIndex(x).getNeuronAt(n).inConnections.get(c).getWeight().value );
					
				} // for each incoming connection
				
			} // for each neuron in layer
			
			
		} // for each layer
		
		
		
	}	

	public static void parseCSVRecord(Vector vec_inputs, int input_counts, Vector vec_outputs, int output_counts, String line) {
		
		String[] parts = line.split(",");
		for ( int x = 0 ; x < parts.length; x++ ) {
			
			if (x < input_counts) {
				vec_inputs.set(x, Double.parseDouble(parts[x]));
			} else if ( x >= input_counts && x < output_counts + input_counts) {
				vec_outputs.set(x - input_counts, Double.parseDouble(parts[x]));
			}
			
			
		}
		
	}	
	
}
