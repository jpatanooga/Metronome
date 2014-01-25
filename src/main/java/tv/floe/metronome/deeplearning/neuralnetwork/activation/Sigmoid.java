package tv.floe.metronome.deeplearning.neuralnetwork.activation;

import org.apache.mahout.math.Matrix;

import tv.floe.metronome.math.MatrixUtils;


public class Sigmoid implements ActivationFunction {

	private static final long serialVersionUID = -6280602270833101092L;

	public Matrix apply(Matrix arg0) {
		return MatrixUtils.sigmoid(arg0);
	}

	@Override
	public Matrix applyDerivative(Matrix input) {
		return MatrixUtils.sigmoid( input ).times( MatrixUtils.oneMinus( MatrixUtils.sigmoid( input ) ) );
	}

}
