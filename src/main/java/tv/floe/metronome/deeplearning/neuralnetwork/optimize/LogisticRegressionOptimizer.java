package tv.floe.metronome.deeplearning.neuralnetwork.optimize;

import java.io.Serializable;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;


import tv.floe.metronome.deeplearning.neuralnetwork.core.LogisticRegression;
import tv.floe.metronome.deeplearning.neuralnetwork.gradient.LogisticRegressionGradient;
import tv.floe.metronome.math.ArrayUtils;
import tv.floe.metronome.math.MatrixUtils;

import cc.mallet.optimize.Optimizable;

public class LogisticRegressionOptimizer implements Optimizable.ByGradientValue,OptimizableByGradientValueMatrix {

	private LogisticRegression logReg;
	private double lr;
	
	
	
	public LogisticRegressionOptimizer(LogisticRegression logReg, double lr) {
		super();
		this.logReg = logReg;
		this.lr = lr;
	}

	@Override
	public int getNumParameters() {
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
		
		if ( index >= MatrixUtils.length(logReg.connectionWeights)) {
		
			return MatrixUtils.getElement( logReg.biasTerms, index - MatrixUtils.length(logReg.connectionWeights) );
			
		}
		
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
		
		if (index >= MatrixUtils.length( logReg.connectionWeights )) {
			
			MatrixUtils.setElement( logReg.biasTerms, index - MatrixUtils.length(logReg.connectionWeights), value);
			
		} else {
			
			MatrixUtils.setElement( logReg.connectionWeights, index, value);
			
		}
		
	}

	@Override
	public void getValueGradient(double[] buffer) {
		
		LogisticRegressionGradient grad = logReg.getGradient( lr );
		
		for (int i = 0; i < buffer.length; i++) {
		
			if ( i < MatrixUtils.length( logReg.connectionWeights )) {
				
				buffer[ i ] = MatrixUtils.getElement( grad.getwGradient(), i ); 
				
			} else {
				
				buffer[ i ] = MatrixUtils.getElement( grad.getbGradient(), i - MatrixUtils.length( logReg.connectionWeights ) );
				
			}
			
		}
		
	}

	@Override
	public double getValue() {
	
		return -logReg.negativeLogLikelihood();

	}

	@Override
	public Matrix getParameters() {
		
		Matrix params = new DenseMatrix(1, getNumParameters() );
		
		for (int i = 0; i < MatrixUtils.length( params ); i++) {
			
		//	params.put(i,getParameter(i));
			MatrixUtils.setElement( params, i, this.getParameter( i ) );
			
		}
		
		return params;
		
	}

	@Override
	public void setParameters(Matrix params) {
		
		//this.setParameters(params.toArray());
		this.setParameters( ArrayUtils.flatten( MatrixUtils.fromMatrix( params ) ) );
		
	}

	@Override
	public Matrix getValueGradient() {
		
		LogisticRegressionGradient grad = logReg.getGradient( lr );
		Matrix ret = new DenseMatrix( 1, getNumParameters() );
		
		for (int i = 0; i < MatrixUtils.length( ret ); i++) {
			
			if ( i < MatrixUtils.length( logReg.connectionWeights  ) ) {
				
				MatrixUtils.setElement( ret, i, MatrixUtils.getElement( grad.getwGradient(), i ) );
				
			} else {
				
				MatrixUtils.setElement( ret, i, MatrixUtils.getElement( grad.getbGradient(), i - MatrixUtils.length( logReg.connectionWeights ) ) );
				
			}
			
		}
		return ret;
	}
	
	
}
