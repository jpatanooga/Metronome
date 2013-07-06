package tv.floe.metronome.classification.neuralnetworks.utils;

import org.apache.mahout.math.Vector;

public class Utils {

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
