package tv.floe.metronome.clustering.kmeans.ir;

import java.io.IOException;
import java.nio.ByteBuffer;

import org.apache.hadoop.io.DataInputBuffer;
import org.apache.hadoop.io.DataOutputBuffer;

import com.cloudera.iterativereduce.Updateable;
import tv.floe.metronome.clustering.kmeans.Means;

public class UpdateableMeans implements Updateable<Means>{

	private Means means = new Means();
	
	@Override
	public ByteBuffer toBytes() {
		DataOutputBuffer buffer = new DataOutputBuffer();
		try {
			means.write(buffer);
		} catch (IOException e) {
			throw new IllegalStateException(e);
		}
		ByteBuffer bb = ByteBuffer.wrap(buffer.getData(), 0, buffer.getLength());
		bb.rewind();
		return bb;
	}

	@Override
	public void fromBytes(ByteBuffer b) {
		byte [] ba = new byte[b.limit() - b.position()];
		b.get(ba);
		DataInputBuffer in = new DataInputBuffer();
		in.reset(ba, ba.length);
		means = new Means();
		try {
			means.readFields(in);
		} catch (IOException e) {
			throw new IllegalStateException(e);
		}
	}

	@Override
	public void fromString(String s) {
		throw new UnsupportedOperationException();
	}

	@Override
	public Means get() {
		return means;
	}

	@Override
	public int hashCode() {
		return means.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		UpdateableMeans other = (UpdateableMeans) obj;
		return means.equals(other.means);
	}

	@Override
	public void set(Means means) {
		this.means = means;
	}
	
	public String toString() {
		return means.toString();
	}

}
