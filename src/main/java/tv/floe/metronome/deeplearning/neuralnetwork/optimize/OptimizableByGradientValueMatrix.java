package tv.floe.metronome.deeplearning.neuralnetwork.optimize;

import org.apache.mahout.math.Matrix;


public interface OptimizableByGradientValueMatrix {

	public int getNumParameters ();

	public Matrix getParameters ();
	public double getParameter (int index);

	public void setParameters (Matrix params);
	public void setParameter (int index, double value);

	public Matrix getValueGradient ();
	public double getValue ();
}
