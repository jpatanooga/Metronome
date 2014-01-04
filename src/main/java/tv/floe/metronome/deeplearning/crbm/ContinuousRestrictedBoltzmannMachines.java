package tv.floe.metronome.deeplearning.crbm;



import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.rbm.RestrictedBoltzmannMachine;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

public class ContinuousRestrictedBoltzmannMachines extends RestrictedBoltzmannMachine {

	public ContinuousRestrictedBoltzmannMachines(int numVisibleNeurons,
			int numHiddenNeurons, RandomGenerator rnd) {
		super(numVisibleNeurons, numHiddenNeurons, rnd);
				
	}
	
	@Override
	public Matrix propDown(Matrix hidden) {
		//return h.mmul(W.transpose()).addRowVector(vBias);		
		return MatrixUtils.addRowVector(hidden.times(this.connectionWeights.transpose()), this.visibleBiasNeurons.viewRow(0)); 
	}
	
	@Override
	public Pair<Matrix, Matrix> sampleVisibleGivenHidden(Matrix hidden) {
		
		Matrix visibleProb = this.propDown(hidden);

		Matrix visibleProbNegExp = visibleProb.clone();
		MatrixUtils.neg(visibleProbNegExp);
		MatrixUtils.exp(visibleProbNegExp);
		
		Matrix visibleProbExp = visibleProb.clone();
		MatrixUtils.exp(visibleProbExp);

		// DoubleMatrix v1Mean = oneDiv(oneMinus(en).sub(oneDiv(aH)));
		
		return null;
	}
	
	

	
	
}
