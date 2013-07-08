package tv.floe.metronome.classification.neuralnetworks.iterativereduce;

import java.nio.ByteBuffer;

import tv.floe.metronome.linearregression.ParameterVector;

import com.cloudera.iterativereduce.Updateable;

public class WeightsUpdateable implements Updateable<ParameterVector> {

	@Override
	public void fromBytes(ByteBuffer arg0) {
		// TODO Auto-generated method stub
		
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
	public ParameterVector get() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void set(ParameterVector arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public ByteBuffer toBytes() {
		// TODO Auto-generated method stub
		return null;
	}

}
