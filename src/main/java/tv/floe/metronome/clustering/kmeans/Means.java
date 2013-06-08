package tv.floe.metronome.clustering.kmeans;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.io.Writable;

public class Means implements Writable, Cloneable, Iterable<Mean> {

	private List<Mean> ma;
	
	public Means() {
		ma = new ArrayList<Mean>();
	}
	
	public Means(Mean ... means) {
		ma = new ArrayList<Mean>();
		for(Mean mean : means) {
			ma.add(mean);
		}
	}
	
	public Means(int n) {
		ma = new ArrayList<Mean>();
		for(int i = 0; i < n; i++) {
			ma.add(new Mean());
		}
	}
	
	public void add(Mean mean) {
		ma.add(mean);
	}
	
	public Mean get(int i) {
		return ma.get(i);
	}
	
	public int size() {
		return ma.size();
	}
	
	public void merge(Means means) {
		if(ma.size() == 0) {
			for(Mean mean : means) {
				ma.add(mean);
			}
			return;
		}
		if(means.size() == 0) {
			return;
		}
		for(int i = 0; i < this.ma.size(); i++) {
			this.get(i).merge(means.get(i));
		}
	}
	
	public void reset() {
		for(Mean mean : ma) {
			mean.reset();
		}
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(ma.size());
		for(Mean mean : ma) {
			mean.write(out);
		}
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		int n = in.readInt();
		ma = new ArrayList<Mean>();
		for(int i = 0; i < n; i++) {
			Mean mean = new Mean();
			mean.readFields(in);
			ma.add(mean);
		}
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ma.hashCode();
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Means other = (Means) obj;
		if (!(ma.equals(other.ma)))
			return false;
		return true;
	}

	public String toString() {
		return ma.toString();
	}

	@Override
	public Iterator<Mean> iterator() {
		return ma.iterator();
	}
	
}
