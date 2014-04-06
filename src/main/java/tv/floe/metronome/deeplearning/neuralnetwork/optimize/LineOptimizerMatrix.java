package tv.floe.metronome.deeplearning.neuralnetwork.optimize;

import org.apache.mahout.math.Matrix;

public interface LineOptimizerMatrix {
	/** Returns the last step size used. */
	public double optimize (Matrix line, double initialStep);

	public interface ByGradient	{
		/** Returns the last step size used. */
		public double optimize (Matrix line, double initialStep);
	}
}
