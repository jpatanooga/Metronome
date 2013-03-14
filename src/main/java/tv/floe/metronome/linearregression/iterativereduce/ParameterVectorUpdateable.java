package tv.floe.metronome.linearregression.iterativereduce;

import java.io.IOException;
import java.nio.ByteBuffer;

import tv.floe.metronome.linearregression.ParameterVector;

import com.cloudera.iterativereduce.Updateable;

public class ParameterVectorUpdateable implements
		Updateable<ParameterVector> {

	ParameterVector param_msg = null;

	public ParameterVectorUpdateable() {
	}

	public ParameterVectorUpdateable(ParameterVector g) {
		this.param_msg = g;
	}

	@Override
	public void fromBytes(ByteBuffer b) {

		b.rewind();

		// System.out.println( " > ParameterVectorGradient::fromBytes > b: " +
		// b.array().length + ", remaining: " + b.remaining() );

		try {
			this.param_msg = new ParameterVector();
			this.param_msg.Deserialize(b.array());
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public ParameterVector get() {
		// TODO Auto-generated method stub
		return this.param_msg;
	}

	@Override
	public void set(ParameterVector t) {
		// TODO Auto-generated method stub
		this.param_msg = t;
	}

	@Override
	public ByteBuffer toBytes() {
		// TODO Auto-generated method stub
		byte[] bytes = null;
		try {
			bytes = this.param_msg.Serialize();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// ByteBuffer buf = ByteBuffer.allocate(bytes.length);
		// buf.put(bytes);
		ByteBuffer buf = ByteBuffer.wrap(bytes);

		return buf;
	}

	@Override
	public void fromString(String s) {
		// TODO Auto-generated method stub

	}
}
