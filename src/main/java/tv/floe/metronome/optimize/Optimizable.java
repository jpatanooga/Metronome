package tv.floe.metronome.optimize;

import java.util.Collection;

//import cc.mallet.optimize.Optimizable;

/**
 * This is copied from MALLET for now until I can figure out how I want to do it
 * 
 * TODO: CHANGE THIS
 * 
 * @author josh
 *
 */
public interface Optimizable {
	public int getNumParameters ();

	public void getParameters (double[] buffer);
	public double getParameter (int index);

	public void setParameters (double[] params);
	public void setParameter (int index, double value);


	public interface ByValue extends Optimizable
	{
		public double getValue ();
	}

	public interface ByGradient extends Optimizable
	{
		public void getValueGradient (double[] buffer);
	}

	public interface ByGradientValue extends Optimizable
	{
		public void getValueGradient (double[] buffer);
		public double getValue ();
	}

	public interface ByHessian extends Optimizable.ByGradientValue
	{
		public void getValueHessian (double[][] buffer);
	}

	public interface ByVotedPerceptron extends Optimizable
	{
		public int getNumInstances ();
		public void getValueGradientForInstance (int instanceIndex, double[] bufffer);
	}

	public interface ByGISUpdate extends Optimizable
	{
		public double getValue();
		public void getGISUpdate (double[] buffer);
	}

	public interface ByBatchGradient extends Optimizable {
		public void getBatchValueGradient (double[] buffer, int batchIndex, int[] batchAssignments);
		public double getBatchValue(int batchIndex, int[] batchAssignments);
	}

	// gsc: for computing gradient from batches in multiple threads
	public interface ByCombiningBatchGradient extends Optimizable {
		public void getBatchValueGradient (double[] buffer, int batchIndex, int[] batchAssignments);
		public double getBatchValue(int batchIndex, int[] batchAssignments);
		public void combineGradients (Collection<double[]> batchGradients, double[] buffer);
		public int getNumBatches();
	}

}
