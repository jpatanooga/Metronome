package tv.floe.metronome.deeplearning.crbm;



import static com.ccc.deeplearning.util.MatrixUtil.log;
import static com.ccc.deeplearning.util.MatrixUtil.oneMinus;
import static com.ccc.deeplearning.util.MatrixUtil.uniform;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;
import org.jblas.DoubleMatrix;

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
		
		// ah
		Matrix visibleProb = this.propDown(hidden);

		// en
		Matrix visibleProbNegExp = visibleProb.clone();
		MatrixUtils.neg(visibleProbNegExp);
		MatrixUtils.exp(visibleProbNegExp);
		
		// ep
		Matrix visibleProbExp = visibleProb.clone();
		MatrixUtils.exp(visibleProbExp);

		// DoubleMatrix v1Mean = oneDiv(oneMinus(en).sub(oneDiv(aH)));		
		Matrix v1Mean = MatrixUtils.oneDiv(MatrixUtils.oneMinus(visibleProbNegExp).minus(MatrixUtils.oneDiv(visibleProb)));
		/*
		Matrix v1Sample = log(
				oneMinus(
				uniform(rng,v1Mean.rows,v1Mean.columns)
				.mul(oneMinus(ep)))
				).div(aH);		
		*/
		//MatrixUtils.lo
		
		return null;
	}
	
	

	
	
}
