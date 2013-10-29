package tv.floe.metronome.classification.neuralnetworks.core;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.input.WeightedSum;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;
import tv.floe.metronome.classification.neuralnetworks.transfer.Tanh;

public class TestNeuralNetwork {

	
	public NeuralNetwork buildXORMLP() throws Exception {
		

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
		//c.parse(null); // default layer: 2-3-2
        c.setConfValue("inputFunction", WeightedSum.class);
		c.setConfValue("transferFunction", Tanh.class);
		c.setConfValue("neuronType", Neuron.class);
		c.setConfValue("networkType", NeuralNetwork.NetworkType.MULTI_LAYER_PERCEPTRON);
		c.setConfValue("layerNeuronCounts", "2,3,1" );
		c.parse(null);
		
		MultiLayerPerceptronNetwork mlp_network = new MultiLayerPerceptronNetwork();
		
		
		
//		int[] neurons = { 2, 3, 1 };
//		c.setLayerNeuronCounts( neurons );
		
		mlp_network.buildFromConf(c);		
		
		return mlp_network;
	}	
	
	
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
	
	@Test
	public void testSerdeMechanics() throws Exception {
		
		NeuralNetwork nn0 = buildXORMLP();
		
		//System.out.println( ">last layer: " + nn0.getLayerByIndex(2).getNeuronsCount() );
		
		byte[] nn_bytes = nn0.Serialize();
		NeuralNetwork nn_serde = NeuralNetwork.Deserialize(nn_bytes);

		// should be a { 2, 3, 1 } network
		
		// check total number layers
		assertEquals( 3, nn_serde.getLayersCount() );
		
		// check layer-neuron counts
		assertEquals( 2, nn_serde.getLayerByIndex(0).getNeuronsCount() );
		assertEquals( 3, nn_serde.getLayerByIndex(1).getNeuronsCount() );
		assertEquals( 1, nn_serde.getLayerByIndex(2).getNeuronsCount() );
		
		
		
		
	}
	
	
	
	
	
	
	
}
