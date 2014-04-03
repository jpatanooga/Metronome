package tv.floe.metronome.deeplearning.neuralnetwork.optimize;

import java.io.Serializable;

import tv.floe.metronome.deeplearning.neuralnetwork.core.LogisticRegression;
import tv.floe.metronome.deeplearning.neuralnetwork.gradient.LogisticRegressionGradient;
import tv.floe.metronome.math.MatrixUtils;

import cc.mallet.optimize.Optimizable;

public class LogisticRegressionOptimizer implements Optimizable.ByGradientValue,Serializable {

	private LogisticRegression logReg;
	private double lr;
	
	
	
	public LogisticRegressionOptimizer(LogisticRegression logReg, double lr) {
		super();
		this.logReg = logReg;
		this.lr = lr;
	}

	@Override
	public int getNumParameters() {
		//return logReg.connectionWeights.length + logReg.biasTerms.length;
		return MatrixUtils.length( logReg.connectionWeights ) + MatrixUtils.length( logReg.biasTerms );
	}

	@Override
	public void getParameters(double[] buffer) {
		for(int i = 0; i < buffer.length; i++) {
			buffer[i] = getParameter(i);
		}

		


	}

	@Override
	public double getParameter(int index) {
		
		//if (index >= logReg.W.length) {
		if ( index >= MatrixUtils.length(logReg.connectionWeights)) {
		
			//return logReg.b.get(index - logReg.W.length);
			return MatrixUtils.getElement( logReg.biasTerms, index - MatrixUtils.length(logReg.connectionWeights) );
			
		}
		
		//return logReg.W.get(index);
		return MatrixUtils.getElement( logReg.connectionWeights , index);
		
	}

	@Override
	public void setParameters(double[] params) {
		
		for(int i = 0; i < params.length; i++) {
			
			setParameter(i,params[i]);
			
		}
		
	}

	@Override
	public void setParameter(int index, double value) {
		
		//if (index >= logReg.W.length) {
		if (index >= MatrixUtils.length( logReg.connectionWeights )) {
			
			// logReg.b.put(index - logReg.W.length,value);
			MatrixUtils.setElement( logReg.biasTerms, index - MatrixUtils.length(logReg.connectionWeights), value);
			
		} else {
			
//			logReg.W.put(index,value);
			MatrixUtils.setElement( logReg.connectionWeights, index, value);
			
		}
		
	}

	@Override
	public void getValueGradient(double[] buffer) {
		
		LogisticRegressionGradient grad = logReg.getGradient( lr );
		
		
		
		for (int i = 0; i < buffer.length; i++) {
		
			//if (i < logReg.W.length) {
			if ( i < MatrixUtils.length( logReg.connectionWeights )) {
				
				//buffer[i] = grad.getwGradient().get(i);
				buffer[ i ] = MatrixUtils.getElement( grad.getwGradient(), i ); 
				
			} else {
				
				//buffer[i] = grad.getbGradient().get(i - logReg.W.length);
				buffer[ i ] = MatrixUtils.getElement( grad.getbGradient(), i - MatrixUtils.length( logReg.connectionWeights ) );
				
			}
			
		}
		
	}

	@Override
	public double getValue() {
	
		return -logReg.negativeLogLikelihood();

	}

}
