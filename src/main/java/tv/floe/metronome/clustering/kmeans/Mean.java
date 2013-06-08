package tv.floe.metronome.clustering.kmeans;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

public class Mean implements Writable, Cloneable {

	private MutablePoint sum = new MutablePoint();
	private int count = 0;
	
	public Mean() {}
	
	public Mean(Point ... points) {
		for(Point point : points) {
			add(point);
		}
	}

	public void reset() {
		sum = null;
		count = 0;
	}
	
	public void add(Point p) {
		if(count == 0) {
			sum = new MutablePoint(p);
			count = 1;
			return;
		}
		for(int i = 0; i < sum.dimensionality(); i++) {
			sum.set(sum.get(i) + p.get(i), i);
		}
		count++;
	}

	public Point toPoint() {
		if(sum.dimensionality() == 0) {
			throw new IllegalStateException();
		}
		double [] averages = new double[sum.dimensionality()];
		for(int i = 0; i < sum.dimensionality(); i++) {
			averages[i] = sum.get(i) / count;
		}
		return new Point(averages);
	}

	public void merge(Mean mean) {
		if(mean.count == 0) {
			return;
		}
		if(count == 0 && mean.count != 0) {
			count = mean.count;
			sum = new MutablePoint(mean.sum);
			return;
		}
		for(int i = 0; i < sum.dimensionality(); i++) {
			sum.set(sum.get(i) + mean.sum.get(i), i);
		}
		count += mean.count;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(count);
		if(count > 0)
			sum.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		count = in.readInt();
		if(sum == null) {
			sum = new MutablePoint();
		}
		if(count>0)
			sum.readFields(in);
	}
	
	@Override
	public int hashCode() {
		return toPoint().hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Mean other = (Mean) obj;
		return toPoint().equals(other.toPoint());
	}

	public String toString() {
		return sum.toString() + " / " + count;
	}

}
