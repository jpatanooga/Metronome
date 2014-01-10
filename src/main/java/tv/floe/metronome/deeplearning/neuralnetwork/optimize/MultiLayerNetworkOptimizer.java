package tv.floe.metronome.deeplearning.neuralnetwork.optimize;

import cc.mallet.optimize.InvalidOptimizableException;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.OptimizationException;
import cc.mallet.optimize.Optimizer;

import java.io.Serializable;

public abstract class MultiLayerNetworkOptimizer implements Optimizable.ByGradientValue,Serializable {

	@Override
	public int getNumParameters() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getParameter(int arg0) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void getParameters(double[] arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setParameter(int arg0, double arg1) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setParameters(double[] arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double getValue() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void getValueGradient(double[] arg0) {
		// TODO Auto-generated method stub
		
	}


}
