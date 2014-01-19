package tv.floe.metronome.deeplearning.rbm;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import org.junit.Test;

import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

public class TestRestrictedBoltzmannMachine {

	public static Matrix buildTestInputDataset() {
		
		double[][] data = new double[][]
				{
					{1,1,1,0,0,0},
					{1,0,1,0,0,0},
					{1,1,1,0,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,1,0}
				};
		
		Matrix d = new DenseMatrix(data);

		return d;
		
	}
	

	
	@Test 
	public void testMatrixSizingOnInputTimes() {
		
//		DoubleMatrix inputTimesPhSample =  this.input.transpose().mmul(ph.getSecond());
// TODO: look at how the training dataset x hiddenSample works out wrt matrix sizes
//		Matrix trainingDataTimesHiddenStates = this.trainingDataset.transpose().times(hidden_sample_init);
		
		
		Matrix input = buildTestInputDataset();
		
		Matrix c = input.clone();
		MatrixUtils.debug_print(c);
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null); 
		
		Pair<Matrix, Matrix> hiddenProbsAndSample = rbm.sampleHiddenGivenVisible( input );

		Matrix hidden_sample_init = hiddenProbsAndSample.getSecond();
		
		//System.out.println( "hidden_sample_init size: " + hidden_sample_init.numRows() + " x " + hidden_sample_init.numCols() );
		//MatrixUtils.debug_print(hidden_sample_init);
		
		Matrix out = input.transpose().times(hidden_sample_init);
		
		//System.out.println( "input size: " + input.transpose().numRows() + " x " + input.transpose().numCols() );
		
		
		
		//MatrixUtils.debug_print( out );
		
	}
	
	/**
	 * Will return a Matrix of size [ inputRowCount, HiddenNeuronCount ]
	 * 
	 * Generates a set of probabilites for hidden states for input sample
	 * 
	 */
	@Test
	public void testPropUp() {
		
		Matrix input = buildTestInputDataset();
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null); 

		
		/**
		 * For every single row we get the 2 hidden states in the "hidden" matrix
		 * 
		 */
		
		for (int x = 0; x < 5000; x++) {
			rbm.contrastiveDivergence(0.1, 1, input);
		}

		Matrix hidden = rbm.propUp(input);
		Matrix recon = rbm.reconstructVisibleInput(input);

		
		//MatrixUtils.debug_print( hidden );
		
		assertEquals( 2, hidden.numCols() );
		assertEquals( 7, hidden.numRows() );
		
		MatrixUtils.debug_print(recon);
		
	}
	
	@Test
	public void testPropDown() {
		
		Matrix input = buildTestInputDataset();
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null); 

		Matrix hidden = rbm.propUp(input);
		
		/**
		 * For every single row we get the 2 hidden states in the "hidden" matrix
		 * 
		 */
		
//		MatrixUtils.debug_print( hidden );
		
		Matrix visRecon = rbm.propDown(hidden);
		
		MatrixUtils.debug_print_matrix_stats(visRecon, "visRecon");
		
		
		
	}
	

	/**
	 * Tests to see if the Cross Entropy drops below a certain level after 1000 epochs
	 * 
	 */
	@Test
	public void testCrossEntropyReconstruction() {
		
		Matrix input = buildTestInputDataset();
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null);
		
		double ce = 0;
		
		for (int x = 0; x < 1000; x++) {
			rbm.contrastiveDivergence(0.1, 1, input);

			ce = rbm.getReConstructionCrossEntropy();
			
			System.out.println("ce: " + ce);
		
		}

		Matrix v = new DenseMatrix(new double[][]
				{
					{1, 1, 1, 0, 0, 0},
				}
		);	

		Matrix recon = rbm.reconstructVisibleInput(v);
		
		// "get the cross entropy somewhere near 0.3 and we're good"
		assertEquals(0.4, ce, 0.2 );
/*
		MatrixUtils.debug_print( rbm.hiddenBiasNeurons );
		
		Vector vec = rbm.hiddenBiasNeurons.viewRow(0);
		
		for ( int x = 0; x < vec.size(); x++ ) {
			System.out.println( vec.get(x) );
		}
		
		rbm.trainingDataset.assign(0);
		
		Matrix preSigmoid = rbm.trainingDataset.times( rbm.connectionWeights );
		preSigmoid = MatrixUtils.addRowVector(preSigmoid, rbm.hiddenBiasNeurons.viewRow(0));
		
		MatrixUtils.debug_print( preSigmoid );
	*/	
		
	}
	
	/**
	 * Tests to see if the optimizer based RBM-contrastive divergence is working 
	 * 
	 */
	@Test
	public void testTrainTilConvergenceOptimizer() {
		
		//trainTillConvergence
		
		Matrix input = buildTestInputDataset();
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null);
		
		double ce = 0;
		

		rbm.trainTillConvergence(0.1, 1, input);
		
	}
	
	
	
	
	
}
