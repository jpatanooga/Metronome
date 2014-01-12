package tv.floe.metronome.deeplearning.rbm;


import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.rbm.RestrictedBoltzmannMachine;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

/**
 * Continuous Restricted Boltzmann Machine
 * 
 * - similar to RBM, but can recognize continuous values as opposed to only binary values
 * 
 * @author josh
 *
 */
public class ContinuousRestrictedBoltzmannMachine extends RestrictedBoltzmannMachine {

/*	
	public ContinuousRestrictedBoltzmannMachine() { 
		
	}
	*/
	public ContinuousRestrictedBoltzmannMachine(Matrix input, int nVisible, int nHidden, Matrix weights,
			Matrix hBias, Matrix vBias, RandomGenerator rng) {
		
		super(input, nVisible, nHidden, weights, hBias, vBias, rng);
		
	}



	public ContinuousRestrictedBoltzmannMachine(int n_visible, int n_hidden, Matrix weights,
			Matrix hbias, Matrix vbias, RandomGenerator rng) {
		
		super(n_visible, n_hidden, weights, hbias, vbias, rng);
		
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
				
		return new Pair<Matrix, Matrix>(v1Mean, v1Sample);
	}
	
	

	
	
}
