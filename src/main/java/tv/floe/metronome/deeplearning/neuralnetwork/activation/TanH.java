package tv.floe.metronome.deeplearning.neuralnetwork.activation;

import java.io.Serializable;

import org.apache.mahout.math.Matrix;

import tv.floe.metronome.math.MatrixUtils;

public class TanH implements ActivationFunction,Serializable {

	@Override
	public Matrix apply(Matrix arg0) {
		// TODO: implement TanH in MatrixUtils
		return MatrixUtils.tanh(arg0);
	}

	@Override
	public Matrix applyDerivative(Matrix input) {
		//1 - tanh^2 x
		//return oneMinus(pow(tanh(input),2));
		return MatrixUtils.oneMinus( MatrixUtils.pow( MatrixUtils.tanh( input ) , 2) );
	}

}
