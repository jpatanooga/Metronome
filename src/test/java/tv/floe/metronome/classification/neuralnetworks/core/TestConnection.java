package tv.floe.metronome.classification.neuralnetworks.core;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.core.neurons.InputNeuron;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class TestConnection {


	Neuron input_layer_neuron_0 = new InputNeuron();
	Neuron input_layer_neuron_1 = new InputNeuron();
	Neuron input_layer_neuron_2 = new InputNeuron();
	
	
	Neuron middle_layer_neuron_0 = new Neuron();
	Neuron middle_layer_neuron_1 = new Neuron();
	Neuron middle_layer_neuron_2 = new Neuron();
	
	Neuron middle_layer_neuron_3 = new Neuron();
	
	
	@Before
	public void setupNeurons() throws Exception {

		
		System.out.println( "Setting up Layers for testing... " + middle_layer_neuron_0.getInConnections().size() );

		middle_layer_neuron_0.addInConnection(input_layer_neuron_0, 0.5);
		
		middle_layer_neuron_1.addInConnection(input_layer_neuron_0, 1.0);
		middle_layer_neuron_1.addInConnection(input_layer_neuron_1, 1.0);
		
		middle_layer_neuron_2.addInConnection(input_layer_neuron_0, 1.0);
		middle_layer_neuron_2.addInConnection(input_layer_neuron_1, 1.0);
		middle_layer_neuron_2.addInConnection(input_layer_neuron_2, 1.0);
		
		
		
		
		input_layer_neuron_0.setInput(0.2d);
		input_layer_neuron_1.setInput(0.2d);
		
		input_layer_neuron_2.setInput(0.3d);
		
		//System.out.println("s-end: " + input_layer_neuron_0.getNetInput());
		
	}
	
	@Test
	public void testGetWeightedInput() throws Exception {

		//setupNeurons();
		
		Connection conn_0 = middle_layer_neuron_0.getInConnections().get(0);

		input_layer_neuron_0.calcOutput();
		
		// need output calc'd
		double wInput = conn_0.getWeightedInput();
		
		assertEquals( 0.1d, wInput, 0.000000d );
		
		
	}

	@Test
	public void testSetToNeuron() throws Exception {

//		middle_layer_neuron_1.addInConnection(input_layer_neuron_0, 1.0);
	/*
		Connection conn_0 = middle_layer_neuron_0.getInConnections().get(0);
		
		// connect to: middle_layer_neuron_3
		
		conn_0.setToNeuron(middle_layer_neuron_3);
		
		Connection c_test_0 = middle_layer_neuron_3.getInConnections().get(0);
		
		assertEquals( conn_0, c_test_0 );
		*/
		
	}
	
	@Test
	public void testSetFromNeuron() {

	}
	
}
