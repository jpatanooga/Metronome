package tv.floe.metronome.classification.neuralnetworks.input;


import static org.junit.Assert.*;


import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.core.neurons.InputNeuron;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class TestWeightedSum {

	@Test
	public void testCalculate() throws Exception {
		
		
		Neuron input_layer_neuron_0 = new InputNeuron();
		Neuron input_layer_neuron_1 = new InputNeuron();
		Neuron input_layer_neuron_2 = new InputNeuron();
		
		
		Neuron middle_layer_neuron_0 = new Neuron();
		Neuron middle_layer_neuron_1 = new Neuron();
		Neuron middle_layer_neuron_2 = new Neuron();
		

		middle_layer_neuron_0.addInConnection(input_layer_neuron_0, 1.0);
		
		middle_layer_neuron_1.addInConnection(input_layer_neuron_0, 1.0);
		middle_layer_neuron_1.addInConnection(input_layer_neuron_1, 1.0);
		
		middle_layer_neuron_2.addInConnection(input_layer_neuron_0, 1.0);
		middle_layer_neuron_2.addInConnection(input_layer_neuron_1, 1.0);
		middle_layer_neuron_2.addInConnection(input_layer_neuron_2, 1.0);
		
		
		
		
		input_layer_neuron_0.setInput(0.2d);
		input_layer_neuron_1.setInput(0.2d);
		
		input_layer_neuron_2.setInput(0.3d);
		
		
/*		
		
		assertEquals( 0.2f, input_layer_neuron_0.getNetInput(), 0.0001f );
		
		input_layer_neuron_0.calcOutput();
		double n0_out = input_layer_neuron_0.getOutput();
		
		assertEquals(0.2f, n0_out, 0.000001f);
*/
		// ------ test: Middle Layer > Neuron 1 ----------
		
		input_layer_neuron_0.calcOutput();
		input_layer_neuron_1.calcOutput();
		
//		System.out.println("\n\n\n> CalcOutput --------");
		middle_layer_neuron_1.calcOutput();
		double n1_1_out = middle_layer_neuron_1.getOutput();
		
//		System.out.println("> CalcOutput --------\n\n");
		assertEquals(0.4f, n1_1_out, 0.000001f);
		
		System.out.println("out: " + n1_1_out );

		// ------ test: Middle Layer > Neuron 2 ----------
		
		input_layer_neuron_2.calcOutput();

		middle_layer_neuron_2.calcOutput();
		double n2_1_out = middle_layer_neuron_2.getOutput();
		
//		System.out.println("> CalcOutput --------\n\n");
		assertEquals(0.7d, n2_1_out, 0.000001f);
		
		System.out.println("out: " + n2_1_out );
		
	}
	
	
	
}
