package tv.floe.metronome.clustering.kmeans;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.io.Writable;

public class Point implements Writable {
	
	protected double [] data = new double[0];

	public Point() {}
	
	public Point(String s) {
		String [] parts = s.split(",");
		double [] data = new double[parts.length];
		for(int i = 0; i < parts.length; i++) {
			data[i] = Double.parseDouble(parts[i]);
		}
		this.data = data;
	}
	
	public Point(double... data) {
		this.data = data.clone();
	}
	
	public int dimensionality() {
		return data.length;
	}
	
	protected void rangeCheck(int i) {
		if(i < 0 || i >= dimensionality()) {
			throw new IllegalArgumentException();
		}
	}
	
	public double get(int i) {
		rangeCheck(i);
		return data[i];
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(data);
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
		Point other = (Point) obj;
		if (!Arrays.equals(data, other.data))
			return false;
		return true;
	}
	
	public String toString() {
		return Arrays.toString(data);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(data.length);
		for(double d : data) {
			out.writeDouble(d);
		}
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		int n = in.readInt();
		data = new double[n];
		for(int i = 0; i < n; i++) {
			data[i] = in.readDouble();
		}
	}
	
}