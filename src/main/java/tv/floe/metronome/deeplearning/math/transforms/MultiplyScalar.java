package tv.floe.metronome.deeplearning.math.transforms;

import org.apache.mahout.math.Matrix;


public class MultiplyScalar extends ScalarMatrixTransform {

	private static final long serialVersionUID = -4947929703315561178L;

	public MultiplyScalar(double scaleBy) {
		super(scaleBy);
	}

	@Override
	public Matrix apply(Matrix input) {
		return input.times(scaleBy);
	}

}
