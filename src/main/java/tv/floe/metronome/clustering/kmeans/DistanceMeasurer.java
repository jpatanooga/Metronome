package tv.floe.metronome.clustering.kmeans;

public class DistanceMeasurer {

	public double distance(Point a, Point b) {
		if(a.dimensionality() != b.dimensionality()) {
			throw new IllegalArgumentException("Can only compare points of the same dimensionality");
		}
		double sum = 0;
		for(int i = 0; i < a.dimensionality(); i++) {
			sum += Math.pow(a.get(i) - b.get(i), 2);
		}
		return Math.sqrt(sum);
	}

}
