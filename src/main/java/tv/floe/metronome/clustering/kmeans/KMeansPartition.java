package tv.floe.metronome.clustering.kmeans;

public class KMeansPartition {
	
	private int desiredMeans;
	
	// These are the current means for comparison.
	private Means currentMeans;
	
	// These are the partial means that we are updating
	private Means updatedMeans = new Means();
	
	// These are the previous means, used for us to determine when a point changes mean.
	private Means previousMeans;
	
	private DistanceMeasurer measurer = new DistanceMeasurer();
	private int changes = 0;
	
	public int getChanges() {
		return changes;
	}
	
	private boolean firstRun;
	
	public KMeansPartition(int desiredMeans) {
		this.desiredMeans = desiredMeans;
		firstRun = true;
	}
	
	public void setMeans(Means means) {
		try{
			means.get(0).toPoint();
		} catch (Exception e) {
			System.out.println("Oops");
		}
		this.previousMeans = this.currentMeans;
		this.currentMeans = means;
		this.changes = 0;
		if(updatedMeans == null) {
			updatedMeans = new Means(means.size());
		} else {
			for(int i = 0; i < updatedMeans.size(); i++) {
				updatedMeans.get(i).reset();
			}
		}
	}
	
	private int closest(Point p, Means means) {
		int index = -1;
		double shortestDistance = Double.MAX_VALUE;
		for(int i = 0; i < means.size(); i++) {
			double distance = measurer.distance(p, means.get(i).toPoint());
			if(distance < shortestDistance) {
				shortestDistance = distance;
				index = i;
			}
		}
		return index;
	}
	
	public void addPoint(Point p) {
		if(firstRun) {
			if(updatedMeans.size() < desiredMeans) {
				Mean mean = new Mean();
				mean.add(p);
				updatedMeans.add(mean);
			}
			return;
		}
		int index = closest(p, currentMeans);
		updatedMeans.get(index).add(p);
		if(previousMeans != null) {
			int previousIndex = closest(p, previousMeans);
			if(index != previousIndex) {
				changes++;
			}
		} else {
			changes++;
		}
	}
	
	public Means getUpdatedMeans() {
		firstRun = false;
		return updatedMeans;
	}
	
}
