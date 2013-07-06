package tv.floe.metronome.classification.neuralnetworks.core;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.core.neurons.InputNeuron;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class TestNeuron {

	Neuron input_layer_neuron_0 = new InputNeuron();
	Neuron input_layer_neuron_1 = new InputNeuron();
	Neuron input_layer_neuron_2 = new InputNeuron();
	
	
	Neuron middle_layer_neuron_0 = new Neuron();
	Neuron middle_layer_neuron_1 = new Neuron();
	Neuron middle_layer_neuron_2 = new Neuron();
	
	@Before
	public void setupNeurons() throws Exception {

		

		middle_layer_neuron_0.addInConnection(input_layer_neuron_0, 1.0);
		
		middle_layer_neuron_1.addInConnection(input_layer_neuron_0, 1.0);
		middle_layer_neuron_1.addInConnection(input_layer_neuron_1, 1.0);
		
		middle_layer_neuron_2.addInConnection(input_layer_neuron_0, 1.0);
		middle_layer_neuron_2.addInConnection(input_layer_neuron_1, 1.0);
		middle_layer_neuron_2.addInConnection(input_layer_neuron_2, 1.0);
		
		
		
		
		input_layer_neuron_0.setInput(0.2d);
		input_layer_neuron_1.setInput(0.2d);
		
		input_layer_neuron_2.setInput(0.3d);
		
		
	}
	
	@Test
	public void testConstructors() {
		
		Neuron n = new Neuron();
		
		Neuron n2 = new InputNeuron();
		
	}
	
	@Test
	public void testAddingConnections() throws Exception {
/*		
		Neuron input_layer_neuron_0 = new InputNeuron();
		Neuron input_layer_neuron_1 = new InputNeuron();
		//Neuron input_layer_neuron_2 = new InputNeuron();
		
		
		Neuron middle_layer_neuron_0 = new Neuron();
		Neuron middle_layer_neuron_1 = new Neuron();
		Neuron middle_layer_neuron_2 = new Neuron();
	*/	
		
		
		
		//Connection c0_0 = new Connection(input_layer_neuron_0, middle_layer_neuron_0);
		//Connection c0_1 = new Connection(input_layer_neuron_0, middle_layer_neuron_1);
		//Connection c0_2 = new Connection(input_layer_neuron_0, middle_layer_neuron_2);

		middle_layer_neuron_0.addInConnection(input_layer_neuron_0, 1.0);
		
		middle_layer_neuron_1.addInConnection(input_layer_neuron_0, 1.0);
		middle_layer_neuron_1.addInConnection(input_layer_neuron_1, 1.0);
		
		middle_layer_neuron_2.addInConnection(input_layer_neuron_0, 1.0);
		middle_layer_neuron_2.addInConnection(input_layer_neuron_1, 1.0);
		
		
		//Connection c1_1 = new Connection(input_layer_neuron_1, middle_layer_neuron_1);
		//Connection c1_2 = new Connection(input_layer_neuron_1, middle_layer_neuron_2);
		
		
		assertEquals(2, middle_layer_neuron_1.getInConnections().size() );
		
		assertEquals(0, input_layer_neuron_0.getInConnections().size() );
		
		assertEquals(3, input_layer_neuron_0.getOutConnections().size() );
		
		assertEquals(2, input_layer_neuron_1.getOutConnections().size() );
		
		assertEquals(0, input_layer_neuron_1.getInConnections().size() );

		//Connection c1 = new Connection();
		
		
		//n.ad
		
	}
	

	@Test
	public void testRandomizeWeights() {
		
		System.out.println("Rndize Weights ----");
		
		int count = middle_layer_neuron_0.getWeights().length;
		
		double w1 = middle_layer_neuron_0.getWeights()[0].getValue();
		
		System.out.println("w1: " + w1);
		
		assertEquals( 1, count  );
		
		middle_layer_neuron_0.randomizeWeights();
		
		count = middle_layer_neuron_0.getWeights().length;
		
		assertEquals( 1, count  );
		
		double w1_rnd = middle_layer_neuron_0.getWeights()[0].getValue();
		
		System.out.println("w1_rnd: " + w1_rnd);
		
		
		
	}
	
	@Test
	public void testRemoveAllConnections() throws Exception {
		
		Neuron l_input_layer_neuron_0 = new InputNeuron();
		Neuron l_input_layer_neuron_1 = new InputNeuron();
		Neuron l_input_layer_neuron_2 = new InputNeuron();
		
		Neuron l_middle_layer_neuron_0 = new Neuron();
		Neuron l_middle_layer_neuron_1 = new Neuron();
		Neuron l_middle_layer_neuron_2 = new Neuron();
		
		
		
		l_middle_layer_neuron_0.addInConnection(l_input_layer_neuron_0, 1.0);
		l_middle_layer_neuron_0.addInConnection(l_input_layer_neuron_1, 1.0);

		l_middle_layer_neuron_1.addInConnection(l_input_layer_neuron_0, 1.0);
		l_middle_layer_neuron_1.addInConnection(l_input_layer_neuron_1, 1.0);
		
		l_middle_layer_neuron_2.addInConnection(l_input_layer_neuron_0, 1.0);
		l_middle_layer_neuron_2.addInConnection(l_input_layer_neuron_1, 1.0);
		l_middle_layer_neuron_2.addInConnection(l_input_layer_neuron_2, 1.0);
		
		assertEquals( 2, l_middle_layer_neuron_0.getInConnections().size() );
		
		// INPUT LAYER
		l_input_layer_neuron_0.removeAllConnections();
		
		assertEquals( 1, l_middle_layer_neuron_0.getInConnections().size() );
		
		assertEquals( 0, l_input_layer_neuron_0.getOutConnections().size() );
		
		System.out.println("in conns: " + l_middle_layer_neuron_2.getInConnections().size() );
		
		l_middle_layer_neuron_2.removeInputConnectionFrom(l_input_layer_neuron_2);
		
		assertEquals( 1, l_middle_layer_neuron_2.getInConnections().size() );
		
	}
	

}
