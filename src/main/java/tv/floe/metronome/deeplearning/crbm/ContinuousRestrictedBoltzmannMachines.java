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
		
		// ah
		Matrix visibleProb = this.propDown(hidden);

		// en
		Matrix visibleProbNegExp = MatrixUtils.exp( MatrixUtils.neg(visibleProb) );
		
		// ep
		Matrix visibleProbExp = MatrixUtils.exp(visibleProb);

		// DoubleMatrix v1Mean = oneDiv(oneMinus(en).sub(oneDiv(aH)));		
		Matrix v1Mean = MatrixUtils.oneDiv(MatrixUtils.oneMinus(visibleProbNegExp).minus(MatrixUtils.oneDiv(visibleProb)));
		/*
		Matrix v1Sample = log(
				oneMinus(
				uniform(rng,v1Mean.rows,v1Mean.columns)
				.mul(oneMinus(ep)))
				).div(aH);		
		*/
		Matrix v1Sample = MatrixUtils.div(
				MatrixUtils.log( 
				MatrixUtils.oneMinus(
						MatrixUtils.uniform(this.randNumGenerator, v1Mean.numRows(), v1Mean.numCols())
								.times(MatrixUtils.oneMinus( visibleProbExp ))
								)
								), visibleProb);
		
		// return new Pair<DoubleMatrix,DoubleMatrix>(v1Mean,v1Sample);
		
		return new Pair<Matrix, Matrix>(v1Mean, v1Sample);
	}
	
	

	
	
}
