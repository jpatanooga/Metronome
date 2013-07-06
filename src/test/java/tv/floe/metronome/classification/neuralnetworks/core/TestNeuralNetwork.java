package tv.floe.metronome.classification.neuralnetworks.core;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.conf.Config;

public class TestNeuralNetwork {

	@Test
	public void testCreateNeuralNetworkByConfig() {

	
	
	}

	@Test
	public void testAddRemoveLayer() {

	
		NeuralNetwork n1 = new NeuralNetwork();
		
		Layer l0 = new Layer(0);
		Layer l1 = new Layer(1);
		Layer l2 = new Layer(1);
		
		n1.addLayer(l0);
		n1.addLayer(l1);
		n1.addLayer(l2);
		
		assertEquals(3, n1.getLayersCount() );
		
		n1.removeLayerAt(0);
		
		assertEquals(2, n1.getLayersCount() );
		
		assertEquals(1, n1.getLayerByIndex(0).getLayerIndex() );
		
	
	}
	
	@Test
	public void testSetGetInputOutputVectors() throws Exception {

		Vector vec = new DenseVector(2);
		vec.set(0, 1);
		vec.set(1,  2);
		
		Config c = new Config();
		c.parse(null);
		
		NeuralNetwork n1 = new NeuralNetwork();
		
		Layer l0 = Layer.createLayer(c, 0);
		Layer l1 = Layer.createLayer(c, 1);
		Layer l2 = Layer.createLayer(c, 2);
		
		n1.addLayer(l0);
		n1.addLayer(l1);
		n1.addLayer(l2);
		
		n1.completeIOWiring();
		
		
		
		n1.setInputVector(vec);
		
		Vector v_out = n1.getOutputVector();
		
		assertEquals( 2, v_out.size() );
		
	
	}
	
	@Test
	public void testCalculate() throws Exception {
		
		Vector vec = new DenseVector(2);
		vec.set(0, 1);
		vec.set(1,  2);
		
		Config c = new Config();
		c.parse(null);
		
		NeuralNetwork n1 = new NeuralNetwork();
		
		Layer l0 = Layer.createLayer(c, 0);
		Layer l1 = Layer.createLayer(c, 1);
		Layer l2 = Layer.createLayer(c, 2);
		
		n1.addLayer(l0);
		n1.addLayer(l1);
		n1.addLayer(l2);
		
		NeuralNetwork.ConnectLayers(l0, l1);
		NeuralNetwork.ConnectLayers(l1, l2);
		
		n1.completeIOWiring();
		
		
		
		n1.setInputVector(vec);
		
		Vector v_out = n1.getOutputVector();
		
		assertEquals( 2, v_out.size() );		

	
	
	}
	
	@Test
	public void testRandomizeWeights() {

	
	
	}
	
	@Test
	public void testSetOutputNeurons() throws Exception {


		Vector vec = new DenseVector(2);
		vec.set(0, 1);
		vec.set(1,  2);
		
		Config c = new Config();
		c.parse(null);
		
		NeuralNetwork n1 = new NeuralNetwork();
		
		Layer l0 = Layer.createLayer(c, 0);
		Layer l1 = Layer.createLayer(c, 1);
		Layer l2 = Layer.createLayer(c, 2);
		
		n1.addLayer(l0);
		n1.addLayer(l1);
		n1.addLayer(l2);
		
		NeuralNetwork.ConnectLayers(l0, l1);
		NeuralNetwork.ConnectLayers(l1, l2);
		
		n1.completeIOWiring();
		
		
		
		n1.setInputVector(vec);
		
		Vector v_out = n1.getOutputVector();
		
		assertEquals( 2, v_out.size() );	
		
		assertEquals( 2, n1.getOutputNeurons().size() );
	
	
	}
	
	@Test
	public void testCreateConnection() throws Exception {


		Vector vec = new DenseVector(2);
		vec.set(0, 1);
		vec.set(1,  2);
		
		Config c = new Config();
		c.parse(null);
		
		NeuralNetwork n1 = new NeuralNetwork();
		
		Layer l0 = Layer.createLayer(c, 0);
		Layer l1 = Layer.createLayer(c, 1);
		Layer l2 = Layer.createLayer(c, 2);
		
		n1.addLayer(l0);
		n1.addLayer(l1);
		n1.addLayer(l2);
		
		NeuralNetwork.ConnectLayers(l0, l1);
		NeuralNetwork.ConnectLayers(l1, l2);
		
		n1.completeIOWiring();
		
		
		
		n1.setInputVector(vec);	
		
		assertEquals( 2, l0.getNeurons().size() );
		
		assertEquals(0, l0.getNeuronAt(0).getInConnections().size() );
		assertEquals(0, l0.getNeuronAt(1).getInConnections().size() );

		assertEquals(3, l0.getNeuronAt(0).getOutConnections().size() );
		assertEquals(3, l0.getNeuronAt(1).getOutConnections().size() );


		assertEquals( 3, l1.getNeurons().size() );
		
		assertEquals(2, l1.getNeuronAt(0).getInConnections().size() );
		assertEquals(2, l1.getNeuronAt(1).getInConnections().size() );
		assertEquals(2, l1.getNeuronAt(2).getInConnections().size() );

		assertEquals(2, l1.getNeuronAt(0).getOutConnections().size() );
		assertEquals(2, l1.getNeuronAt(1).getOutConnections().size() );
		assertEquals(2, l1.getNeuronAt(2).getOutConnections().size() );
		
		assertEquals( 2, l2.getNeurons().size() );
		
		assertEquals(3, l2.getNeuronAt(0).getInConnections().size() );
		assertEquals(3, l2.getNeuronAt(1).getInConnections().size() );

		assertEquals(0, l2.getNeuronAt(0).getOutConnections().size() );
		assertEquals(0, l2.getNeuronAt(1).getOutConnections().size() );
		
		
	}
	
	
	
	
	
	
	
	
}
