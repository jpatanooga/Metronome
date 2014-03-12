package tv.floe.metronome.deeplearning.dbn.iterativereduce;

import java.io.IOException;
import java.nio.ByteBuffer;

//import tv.floe.metronome.linearregression.ParameterVector;

import com.cloudera.iterativereduce.Updateable;

public class ParameterVectorUpdateable implements Updateable<ParameterVector> {
	
	ParameterVector param_msg = null;
	
	public ParameterVectorUpdateable() {
	}
	
	public ParameterVectorUpdateable(ParameterVector g) {
		this.param_msg = g;
	}
	
	@Override
	public void fromBytes(ByteBuffer b) {
		
		b.rewind();
		
		try {
			this.param_msg = new ParameterVector();
			this.param_msg.Deserialize(b.array());
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public ParameterVector get() {
		return this.param_msg;
	}
	
	@Override
	public void set(ParameterVector t) {
		this.param_msg = t;
	}
	
	@Override
	public ByteBuffer toBytes() {
		byte[] bytes = null;
		try {
			bytes = this.param_msg.Serialize();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		ByteBuffer buf = ByteBuffer.wrap(bytes);
		
		return buf;
	}
	
	@Override
	public void fromString(String s) {
	
	}
}
