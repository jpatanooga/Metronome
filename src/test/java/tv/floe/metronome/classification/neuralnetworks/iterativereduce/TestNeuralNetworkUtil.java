package tv.floe.metronome.classification.neuralnetworks.iterativereduce;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.Weight;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.input.WeightedSum;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;
import tv.floe.metronome.classification.neuralnetworks.transfer.Tanh;

public class TestNeuralNetworkUtil {

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
		c.parse(null); // default layer: 2-3-2
        c.setConfValue("inputFunction", WeightedSum.class);
		c.setConfValue("transferFunction", Tanh.class);
		c.setConfValue("neuronType", Neuron.class);
		c.setConfValue("networkType", NeuralNetwork.NetworkType.MULTI_LAYER_PERCEPTRON);
		c.setConfValue("layerNeuronCounts", "2,3,1" );
		
		
		MultiLayerPerceptronNetwork mlp_network = new MultiLayerPerceptronNetwork();
		
		
		
		int[] neurons = { 2, 3, 1 };
		c.setLayerNeuronCounts( neurons );
		
		mlp_network.buildFromConf(c);		
		
		return mlp_network;
	}
	
	@Test
	public void testCollectNetworks() throws Exception {

	
		NeuralNetworkUtil util = new NeuralNetworkUtil();
		
		NeuralNetwork nn0 =  buildXORMLP();
		nn0.getLayerByIndex(1).getNeuronAt(0).getInConnections().get(0).setWeight(new Weight(0));
		nn0.getLayerByIndex(1).getNeuronAt(0).getInConnections().get(1).setWeight(new Weight(1));
		util.AccumulateWorkerNetwork( nn0 );
		
		
		
		
		util.AccumulateWorkerNetwork( buildXORMLP() );
		util.AccumulateWorkerNetwork( buildXORMLP() );
		
		assertEquals( 3, util.getNetworkBufferCount() );
	
	
	}
	
	@Test
	public void testAverageCollectedNetworks() {
		
		
	}

}
