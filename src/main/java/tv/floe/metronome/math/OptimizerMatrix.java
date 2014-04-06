package tv.floe.metronome.math;

public interface OptimizerMatrix {
	
	public boolean optimize ();
	public boolean optimize (int numIterations);
	public boolean isConverged();
	
}
