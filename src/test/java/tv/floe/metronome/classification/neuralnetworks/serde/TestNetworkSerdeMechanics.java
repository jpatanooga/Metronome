package tv.floe.metronome.classification.neuralnetworks.serde;

import static org.junit.Assert.*;

import java.io.IOException;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.activation.Tanh;
import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.input.WeightedSum;
import tv.floe.metronome.classification.neuralnetworks.iterativereduce.MasterNode;
import tv.floe.metronome.classification.neuralnetworks.iterativereduce.NetworkWeightsUpdateable;
import tv.floe.metronome.classification.neuralnetworks.iterativereduce.NeuralNetworkWeightsDelta;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;

public class TestNetworkSerdeMechanics {

	private static JobConf defaultConf = new JobConf();
	private static FileSystem localFs = null;
	static {
		try {
			defaultConf.set("fs.defaultFS", "file:///");
			localFs = FileSystem.getLocal(defaultConf);
		} catch (IOException e) {
			throw new RuntimeException("init failure", e);
		}
	}
	
	
	@Test
	public void testBaseJavaObjectSerialization() {

		NetworkWeightsUpdateable nwu = new NetworkWeightsUpdateable();
		
		Config c = new Config();
		c.parse(null); // default layer: 2-3-2
        c.setConfValue("inputFunction", WeightedSum.class);
		c.setConfValue("transferFunction", Tanh.class);
		c.setConfValue("neuronType", Neuron.class);
		c.setConfValue("networkType", NeuralNetwork.NetworkType.MULTI_LAYER_PERCEPTRON);
		c.setConfValue("layerNeuronCounts", "2,3,1" );
		c.parse(null);
		
		NeuralNetwork nn = new MultiLayerPerceptronNetwork();
		
		try {
			nn.buildFromConf(c);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		NeuralNetworkWeightsDelta nnwd = new NeuralNetworkWeightsDelta();
		nnwd.network = nn;
		
		nwu.set(nnwd);
		
		nwu.toBytes();
		
		
		
		
		
		
	}
	
	@Test
	public void testMasterNodeObjectSerialization() {

		NetworkWeightsUpdateable nwu = new NetworkWeightsUpdateable();
		
		Config c = new Config();
		c.parse(null); // default layer: 2-3-2
        c.setConfValue("inputFunction", WeightedSum.class);
		c.setConfValue("transferFunction", Tanh.class);
		c.setConfValue("neuronType", Neuron.class);
		c.setConfValue("networkType", NeuralNetwork.NetworkType.MULTI_LAYER_PERCEPTRON);
		c.setConfValue("layerNeuronCounts", "2,3,1" );
		c.parse(null);
		
		NeuralNetwork nn = new MultiLayerPerceptronNetwork();
		
		try {
			nn.buildFromConf(c);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		/*
		NeuralNetworkWeightsDelta nnwd = new NeuralNetworkWeightsDelta();
		nnwd.network = nn;
		
		nwu.set(nnwd);
		
		nwu.toBytes();
		*/
		
		MasterNode mnode = new MasterNode();
		mnode.master_nn = nn;
		
		
		
		try {

			Path out = new Path("/tmp/fooTest.model"); 
			FileSystem fs =
					  out.getFileSystem(defaultConf); 
			
			FSDataOutputStream fos;

			fos = fs.create(out);
			  //LOG.info("Writing master results to " + out.toString());
			  mnode.complete(fos);
			  
			  fos.flush(); 
			  fos.close();

			//BufferedWriter bw = new BufferedWriter(new FileWriter(output_path));
			//master.complete( bw );
			
			//bw.close();
			/*
			FileOutputStream fs = new FileOutputStream(output_path);
		    ObjectOutputStream oos = new ObjectOutputStream(fs);
		    
		    //oos.writeObject( master. );
		    //master.complete(oos);
		    
		    oos.flush();
		    oos.close();
*/
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		  		
		
		
		
	}	

}
