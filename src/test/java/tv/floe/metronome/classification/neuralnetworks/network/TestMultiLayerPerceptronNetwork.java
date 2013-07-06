package tv.floe.metronome.classification.neuralnetworks.network;

import static org.junit.Assert.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Iterator;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;


import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;

public class TestMultiLayerPerceptronNetwork {

	@Test
	public void testBasicConstruction() throws Exception {


		Vector vec = new DenseVector(2);
		vec.set(0, 1);
		vec.set(1,  2);
		
		Config c = new Config();
		c.parse(null); // default layer: 2-3-2
		
		MultiLayerPerceptronNetwork mlp_network = new MultiLayerPerceptronNetwork();
		
		mlp_network.buildFromConf(c);
		

		mlp_network.setInputVector(vec);
		
		//mlp_network.
		
		Layer l0 = mlp_network.getLayerByIndex(0);
		
		assertEquals( 2, l0.getNeurons().size() );
		
		assertEquals(0, l0.getNeuronAt(0).getInConnections().size() );
		assertEquals(0, l0.getNeuronAt(1).getInConnections().size() );

		assertEquals(3, l0.getNeuronAt(0).getOutConnections().size() );
		assertEquals(3, l0.getNeuronAt(1).getOutConnections().size() );

		
		
		Layer l1 = mlp_network.getLayerByIndex(1);

		assertEquals( 3, l1.getNeurons().size() );
		
		assertEquals(2, l1.getNeuronAt(0).getInConnections().size() );
		assertEquals(2, l1.getNeuronAt(1).getInConnections().size() );
		assertEquals(2, l1.getNeuronAt(2).getInConnections().size() );

		assertEquals(2, l1.getNeuronAt(0).getOutConnections().size() );
		assertEquals(2, l1.getNeuronAt(1).getOutConnections().size() );
		assertEquals(2, l1.getNeuronAt(2).getOutConnections().size() );
		
		
		
		Layer l2 = mlp_network.getLayerByIndex(2);
		
		assertEquals( 2, l2.getNeurons().size() );
		
		assertEquals(3, l2.getNeuronAt(0).getInConnections().size() );
		assertEquals(3, l2.getNeuronAt(1).getInConnections().size() );

		assertEquals(0, l2.getNeuronAt(0).getOutConnections().size() );
		assertEquals(0, l2.getNeuronAt(1).getOutConnections().size() );		
		
	}
	
	public void parseCSVRecord(Vector vec_inputs, int input_counts, Vector vec_outputs, int output_counts, String line) {
		
		String[] parts = line.split(",");
		for ( int x = 0 ; x < parts.length; x++ ) {
			//vec_out.set(x, Double.parseDouble(parts[x]));
			
			if (x < input_counts) {
				vec_inputs.set(x, Double.parseDouble(parts[x]));
			} else if ( x >= input_counts && x < output_counts + input_counts) {
				vec_outputs.set(x - input_counts, Double.parseDouble(parts[x]));
			}
			
			
		}
		
		//return Double.parseDouble( parts[6] ); 
		
	}
	
	  
	  public static void PrintVector(Vector v) {
	    
	    boolean first = true;
	    Iterator<Vector.Element> nonZeros = v.iterator();
	    while (nonZeros.hasNext()) {
	      Vector.Element vec_loc = nonZeros.next();
	      
	      if (!first) {
	        System.out.print(",");
	      } else {
	        first = false;
	      }
	      
	      System.out.print(" " + vec_loc.get());
	      
	    }
	    
	    System.out.println("\n");
	    
	  }	
	
	@Test
	public void testIrisDatasetRun() throws Exception {
		
		Config c = new Config();
		c.parse(null); // default layer: 2-3-2
		c.setConfValue("useBiasNeuron", "true");
		
	
		int[] neurons = { 4, 16, 3 };
		c.setLayerNeuronCounts( neurons );
		
		// layers: 4, 16, 3
        MultiLayerPerceptronNetwork neuralNet = new MultiLayerPerceptronNetwork();
        neuralNet.buildFromConf(c);
        
        int num_inputs = 4;
        int num_outputs = 3;
		
		for ( int x = 0; x < 200; x++ ) {
			
			
			BufferedReader reader = new BufferedReader( new FileReader("src/test/resources/data/iris/iris_data_normalised.txt") );
			
			
			String line = reader.readLine();
			
			while (line != null && line.length() > 0) {
	
				//System.out.println(line);
	
				
	
				if (null == line || line.trim().equals("")) {
					
					System.out.println("> bad line > " + line );
					
				} else {
					
	
					Vector vec_inputs = new RandomAccessSparseVector( num_inputs );
					
					Vector vec_outputs = new RandomAccessSparseVector( num_outputs );
					
				    
				    //double actual = factory.processLineAlt(line, vec);
					parseCSVRecord(vec_inputs, num_inputs, vec_outputs, num_outputs, line);

					/*
					PrintVector( vec_inputs );
					PrintVector( vec_outputs );
					
					System.out.println(" ----------------- " );
					*/
					
					assertEquals( num_inputs, vec_inputs.size() );
					
					assertEquals( num_outputs, vec_outputs.size() );
					
					neuralNet.train(vec_outputs, vec_inputs);

				} // if
				
				line = reader.readLine();

			} // while
			
			
			reader = new BufferedReader( new FileReader("src/test/resources/data/iris/iris_data_normalised.txt") );
			
			
			line = reader.readLine();
			
			int total_records = 0;
			int number_correct = 0;
			
			while (line != null && line.length() > 0) {
	
				if (null == line || line.trim().equals("")) {
					
					System.out.println("> bad line > " + line );
					
				} else {
					
					total_records++;
					
					Vector vec_inputs = new RandomAccessSparseVector( num_inputs );
					
					Vector vec_outputs = new RandomAccessSparseVector( num_outputs );
					
					parseCSVRecord(vec_inputs, num_inputs, vec_outputs, num_outputs, line);
					
					
		            neuralNet.setInputVector( vec_inputs );
		            neuralNet.calculate();
		            Vector networkOutput = neuralNet.getOutputVector();
		            
		            //PrintVector( networkOutput );
		            //System.out.println( "Prediction: " + networkOutput.maxValueIndex() );

		            if (networkOutput.maxValueIndex() == 0 && vec_outputs.get(0) == 1.0 || networkOutput.maxValueIndex() == 1 && vec_outputs.get(1) == 1.0  || networkOutput.maxValueIndex() == 2 && vec_outputs.get(2) == 1.0) {
		            	number_correct++;
		            }
		            	
		            	
		            
		            		
		            		

				} // if
	            
	            line = reader.readLine();
	            
			} // while
			
			System.out.println("> Total Records: " + total_records);
			System.out.println("> Correct: " + number_correct);

			
			System.out.println("----------------------- ");			
		} // for
				
		
	}
	

}
