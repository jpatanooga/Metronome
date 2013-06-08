package tv.floe.metronome.clustering.kmeans;

public class MutablePoint extends Point {

	public MutablePoint() {}
	
	public void parse(String s) {
		String [] parts = s.split(",");
		double [] data = new double[parts.length];
		for(int i = 0; i < parts.length; i++) {
			data[i] = Double.parseDouble(parts[i]);
		}
		this.data = data;
	}
	
	public MutablePoint(Point point) {
		this.data = point.data.clone();
	}
	
	public void set(double d, int index) {
		rangeCheck(index);
		data[index] = d;
	}
	
}
