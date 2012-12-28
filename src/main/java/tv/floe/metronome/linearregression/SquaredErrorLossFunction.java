package tv.floe.metronome.linearregression;

import org.apache.mahout.math.Vector;

/**
 * 
 * Loss function for Linear Regression in Stochastic Gradient Descent
 * 
 * 1/2( h(x) - y )^2
 * 
 * TODO:
 * -	add a base interface?
 * 
 * @author josh
 *
 */
public class SquaredErrorLossFunction {
	
	/**
	 * calculates the squared error loss for linear regression
	 * 
	 * hypothesis_value is the dot product of the current parameter vector and the current training instance
	 * 
	 * @param instance
	 * @param actual
	 * @return
	 */
	static public double Calc(double hypothesis_value, double actual) {
		
		double error = 0.0d;
		
		//error = ((hypothesis_value - actual)^2) / 2;
		error = Math.pow(hypothesis_value - actual, 2) / 2;
		return error;
		
	}

}
