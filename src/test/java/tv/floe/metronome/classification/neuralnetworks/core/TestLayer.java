package tv.floe.metronome.classification.neuralnetworks.core;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.InputNeuron;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class TestLayer {

	
	@Test
	public void testCreateLayer() throws Exception {
		
		Layer l0 = new Layer(1);
		
		l0.addNeuron(new Neuron());
		l0.addNeuron(new Neuron());
		l0.addNeuron(new Neuron());
		
		assertEquals(3, l0.getNeuronsCount() );
		
	}
	
	@Test
	public void testCreateLayerViaConfig() throws Exception {
		
		Config c = new Config();
		c.parse(null);
		
		Layer input_layer = Layer.createLayer(c, 0);
		
		Layer middle_layer = Layer.createLayer(c, 1);
		
		Layer output_layer = Layer.createLayer(c, 2);

		assertEquals(2, input_layer.getNeuronsCount());
		assertEquals(3, middle_layer.getNeuronsCount());
		assertEquals(2, output_layer.getNeuronsCount());
		
	}
	
	
	
	@Test
	public void testAddNeurons() throws Exception {
		
		//System.out.println("> testAddNeurons() ");
		
		Config c = new Config();
		
		c.parse(null);
		
		// adds 2 input neurons
		Layer input_layer = Layer.createLayer(c, 0);
		
		Neuron n1 = new InputNeuron();
		
		input_layer.addNeuron(0, n1);
		
		assertEquals(3, input_layer.getNeuronsCount());
		
		Neuron n2 = new Neuron();
		
		try {
		
			input_layer.addNeuron(0, n2);
			
		} catch (Exception e) {
			// not an input neuron
			System.out.println("> " + e.toString());
		}
		try {
			
			input_layer.addNeuron(n2);
			
		} catch (Exception e) {
			// not an input neuron
			System.out.println("> " + e.toString());
		}
		
		assertEquals(3, input_layer.getNeuronsCount());
		
		
	}
/*	
	@Test
	public void testCalculate() throws Exception {
		
		Config c = new Config();
		
		c.parse(null);
		
		// adds 2 input neurons
		Layer input_layer = Layer.createLayer(c, 0);
		
		for (int x = 0; x < input_layer.getNeuronsCount(); x++ ) {
			
			InputNeuron ip = (InputNeuron) input_layer.getNeuronAt(x);
			ip.setInput(1.0);
			
			
		}
		
		
	}
	*/

}
