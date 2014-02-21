package tv.floe.metronome.deeplearning.math.transforms;

import org.apache.mahout.math.Matrix;



public abstract class ScalarMatrixTransform implements MatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2491009087533310977L;
	protected double scaleBy;
	
	public ScalarMatrixTransform(double scaleBy) {
		this.scaleBy = scaleBy;
	}

	@Override
	public abstract Matrix apply(Matrix input);
	
	
}
