package tv.floe.metronome.deeplearning.neuralnetwork.activation;

import java.io.Serializable;

import org.apache.mahout.math.Matrix;

import tv.floe.metronome.math.MatrixUtils;

public class SoftMax  implements ActivationFunction,Serializable {

	private static final long serialVersionUID = -1820333963272195229L;

	public Matrix apply(Matrix arg0) {
		return MatrixUtils.softmax(arg0);
	}

	@Override
	public Matrix applyDerivative(Matrix input) {
		return MatrixUtils.elementWiseMultiplication( MatrixUtils.softmax( input ), MatrixUtils.oneMinus( MatrixUtils.softmax( input ) ) );
	}

}
