package tv.floe.metronome.classification.neuralnetworks.network;

import static org.junit.Assert.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.activation.Tanh;
import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.input.WeightedSum;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;

public class TestLearnXORFunctionWithMLPN {

	@Test
	public void test() throws Exception {

		
		Vector v0 = new DenseVector(2);
		v0.set(0, 0);
		v0.set(1, 0);
		Vector v0_out = new DenseVector(1);
		v0_out.set(0, 0);
		//xor_recs.add(v0);

		Vector v1 = new DenseVector(2);
		v1.set(0, 0);
		v1.set(1, 1);

		Vector v1_out = new DenseVector(1);
		v1_out.set(0, 1);
		//xor_recs.add(v1);

		
		
		Vector v2 = new DenseVector(2);
		v2.set(0, 1);
		v2.set(1, 0);

		Vector v2_out = new DenseVector(1);
		v2_out.set(0, 1);
		//xor_recs.add(v2);

		
		
		Vector v3 = new DenseVector(2);
		v3.set(0, 1);
		v3.set(1, 1);

		Vector v3_out = new DenseVector(1);
		v3_out.set(0, 0);
		//xor_recs.add(v3);

		
		Config c = new Config();
		c.parse(null); // default layer: 2-3-2
        c.setConfValue("inputFunction", WeightedSum.class);
		c.setConfValue("transferFunction", Tanh.class);
		c.setConfValue("neuronType", Neuron.class);
		c.setConfValue("networkType", NeuralNetwork.NetworkType.MULTI_LAYER_PERCEPTRON);
		c.setConfValue("layerNeuronCounts", "2,3,1" );
		
		
		MultiLayerPerceptronNetwork mlp_network = new MultiLayerPerceptronNetwork();
		
		
		

//		mlp_network.setInputVector(vec);
		
		int[] neurons = { 2, 3, 1 };
		c.setLayerNeuronCounts( neurons );
		
		mlp_network.buildFromConf(c);
        
		
		for ( int x = 0; x < 40000; x++ ) {
			
			
					

			mlp_network.train(v0_out, v0);
			mlp_network.train(v1_out, v1);
			mlp_network.train(v2_out, v2);
			mlp_network.train(v3_out, v3);

			
			int total_records = 0;
			int number_correct = 0;
					
			total_records++;
			
			
			mlp_network.setInputVector( v0 );
			mlp_network.calculate();
            Vector networkOutput = mlp_network.getOutputVector();

            System.out.println( "> out: 0 =? " + networkOutput.get(0) );
		            
            
            
			mlp_network.setInputVector( v1 );
			mlp_network.calculate();
            Vector networkOutput_1 = mlp_network.getOutputVector();

            System.out.println( "> out: 1 =? " + networkOutput_1.get(0) );
		            		

			mlp_network.setInputVector( v2 );
			mlp_network.calculate();
            Vector networkOutput_2 = mlp_network.getOutputVector();

            System.out.println( "> out: 1 =? " + networkOutput_2.get(0) );


			mlp_network.setInputVector( v3 );
			mlp_network.calculate();
            Vector networkOutput_3 = mlp_network.getOutputVector();

            System.out.println( "> out: 0 =? " + networkOutput_3.get(0) );
            
            
			
			//System.out.println("> Total Records: " + total_records);
			//System.out.println("> Correct: " + number_correct);

			
			System.out.println("----------------------- ");			
		} // for
	
	}

}
