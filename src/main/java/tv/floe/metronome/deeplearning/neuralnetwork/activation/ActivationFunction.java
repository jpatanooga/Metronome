package tv.floe.metronome.deeplearning.neuralnetwork.activation;

import org.apache.mahout.math.Matrix;


public interface ActivationFunction {

	/**
	 * Applies the derivative of this function
	 * @param input the input to apply it to
	 * @return the derivative of this function applied to 
	 * the input
	 */
	public Matrix applyDerivative(Matrix input);

}
