package tv.floe.metronome.classification.neuralnetworks.iterativereduce;

import java.io.IOException;
import java.nio.ByteBuffer;

//import tv.floe.metronome.classification.logisticregression.iterativereduce.ParameterVector;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
//import tv.floe.metronome.linearregression.ParameterVector;

import com.cloudera.iterativereduce.Updateable;

public class NetworkWeightsUpdateable implements Updateable<NeuralNetworkWeightsDelta> {

	NeuralNetworkWeightsDelta networkUpdate = null;
	
	public NetworkWeightsUpdateable() { }
	
	public NetworkWeightsUpdateable(NeuralNetworkWeightsDelta nnwd) {
		this.networkUpdate = nnwd;
	}
	
	@Override
	public void fromBytes(ByteBuffer b) {

	    b.rewind();
	    
	    
	    try {
	      this.networkUpdate.Deserialize(b.array());
	    } catch (IOException e) {
	      // TODO Auto-generated catch block
	      e.printStackTrace();
	    }		
	}

	@Override
	public void fromString(String arg0) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * TODO: Not every algorithm has a "parameter vector" ---- we need to re-think this
	 * in the base interface
	 * 
	 */
	@Override
	public NeuralNetworkWeightsDelta get() {
		return this.networkUpdate;
	}

	@Override
	public void set(NeuralNetworkWeightsDelta net) {
		this.networkUpdate = net;
	}

	@Override
	public ByteBuffer toBytes() {
	    byte[] bytes = null;
	    try {
	      bytes = this.networkUpdate.Serialize();
	    } catch (IOException e) {
	      // TODO Auto-generated catch block
	      e.printStackTrace();
	    }
	    
	    ByteBuffer buf = ByteBuffer.wrap(bytes);
	    
	    return buf;	
	}

}
