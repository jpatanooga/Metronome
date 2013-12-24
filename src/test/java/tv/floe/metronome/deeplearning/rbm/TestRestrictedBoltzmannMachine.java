package tv.floe.metronome.deeplearning.rbm;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

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
					{0,0,1,1,1,0}
				};
		
		Matrix d = new DenseMatrix(data);

		return d;
		
	}
	
	/**
	 * Tests functionality for the code based on equation (7) from Hinton
	 * 
	 */
	@Test
	public void testGenerateProbabilitiesForSettingHiddenStatesToOne() {
		
		//System.out.println("------ testGenerateProbabilitiesForSettingHiddenStatesToOne -------");
		
		Matrix input = buildTestInputDataset();
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null); 
		
		Pair<Matrix, Matrix> hiddenProbsAndSample = rbm.sampleHiddenGivenVisible( input );

		Matrix hidden_sample_init = hiddenProbsAndSample.getSecond();
		
		//System.out.println( "hidden_sample_init size: " + hidden_sample_init.numRows() + " x " + hidden_sample_init.numCols() );
		
//		Pair<Matrix, Matrix> visibleProbsAndSample = rbm.gibbsSamplingStepFromHidden(hidden_sample_init);
/*
		Matrix visible_sample = visibleProbsAndSample.getSecond();
		
		System.out.println( "visible_sample size: " + visible_sample.numRows() + " x " + visible_sample.numCols() );
		*/
		//System.out.println("-------------");
		
	}

	/**
	 * Tests functionality for the code based on equation (8) from Hinton
	 * 
	 */
	@Test
	public void testGenerateProbabilitiesForSettingVisibleStatesToOne() {
		
		
		
		
	}
	
	@Test
	public void testContrastiveDivergenceLearning() {
		
		
	}
	
	@Test 
	public void testMatrixSizingOnInputTimes() {
		
//		DoubleMatrix inputTimesPhSample =  this.input.transpose().mmul(ph.getSecond());
// TODO: look at how the training dataset x hiddenSample works out wrt matrix sizes
//		Matrix trainingDataTimesHiddenStates = this.trainingDataset.transpose().times(hidden_sample_init);
		
		
		Matrix input = buildTestInputDataset();
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null); 
		
		Pair<Matrix, Matrix> hiddenProbsAndSample = rbm.sampleHiddenGivenVisible( input );

		Matrix hidden_sample_init = hiddenProbsAndSample.getSecond();
		
		//System.out.println( "hidden_sample_init size: " + hidden_sample_init.numRows() + " x " + hidden_sample_init.numCols() );
		MatrixUtils.debug_print(hidden_sample_init);
		
		Matrix out = input.transpose().times(hidden_sample_init);
		
		//System.out.println( "input size: " + input.transpose().numRows() + " x " + input.transpose().numCols() );
		
		
		
		//MatrixUtils.debug_print( out );
		
	}
	
	/**
	 * Will return a Matrix of size [ inputRowCount, HiddenNeuronCount ]
	 * 
	 */
	@Test
	public void testPropUp() {
		
		Matrix input = buildTestInputDataset();
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null); 

		Matrix hidden = rbm.generateProbabilitiesForHiddenStatesBasedOnVisibleStates(input);
		
		/**
		 * For every single row we get the 2 hidden states in the "hidden" matrix
		 * 
		 */
		
		//MatrixUtils.debug_print( hidden );
		
	}
	
	@Test
	public void testPropDown() {
		
		Matrix input = buildTestInputDataset();
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null); 

		Matrix hidden = rbm.generateProbabilitiesForHiddenStatesBasedOnVisibleStates(input);
		
		/**
		 * For every single row we get the 2 hidden states in the "hidden" matrix
		 * 
		 */
		
//		MatrixUtils.debug_print( hidden );
		
		rbm.generateProbabilitiesForVisibleStatesBasedOnHiddenStates(hidden);
		
		
		
	}
	
	@Test
	public void testBaseCD() {
		
		Matrix input = buildTestInputDataset();
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null);
		
		for (int x = 0; x < 5000; x++) {
			rbm.contrastiveDivergence(1, input);
		}
		
		Matrix v = new DenseMatrix(new double[][]
				{
					{1, 1, 0, 0, 0, 0},
					{0, 0, 0, 1, 1, 0}
				}
		);	


		Matrix recon = rbm.reconstructVisibleInput(v);
		
		MatrixUtils.debug_print(recon);
		
		
		
	}
	
	
	
	
	
}
