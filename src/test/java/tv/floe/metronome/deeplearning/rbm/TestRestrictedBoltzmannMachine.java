package tv.floe.metronome.deeplearning.rbm;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.iterativereduce.iris.IrisDatasetUtils;
import tv.floe.metronome.deeplearning.neuralnetwork.core.LogisticRegression;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

public class TestRestrictedBoltzmannMachine {


	double[][] xor_input = new double[][] 
	{
			{0,0},
			{0,1},
			{1,0},
			{1,1}
			
	};
	
	double[][] xor_labels = new double[][] 
	{
			{1, 0},
			{0, 1},
			{0, 1},
			{1, 0}
	};
		
	Matrix x_xor_Matrix = new DenseMatrix(xor_input);	
	
	
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
		
		MatrixUtils.debug_print(input);
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
	 * 
	 * Looks at how well CDk drops in an SGD fashion
	 * 
	 */
	@Test
	public void testBaseCDk_NonCG() {
		
		//Matrix input = buildTestInputDataset();
		
		double[][] data_simple = new double[][]
				{
					{1,1,1,0,0,0},
					{0,0,0,1,1,1}
/*					{1,1,1,0,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,1,0}
					*/
				};
		
		Matrix input = new DenseMatrix(data_simple);		
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null);
		
//		MatrixUtils.debug_print( rbm.connectionWeights );
//		MatrixUtils.debug_print( rbm.visibleBiasNeurons );
//		MatrixUtils.debug_print( rbm.hiddenBiasNeurons );
		
		
		double ce = 0;
		
		for (int x = 0; x < 1000; x++) {
			
			rbm.contrastiveDivergence(0.001, 5, input);
			//rbm.trainTillConvergence(0.001, 1, input);

			ce = rbm.getReConstructionCrossEntropy();
			
			System.out.println("ce: " + ce);
			
		
		}
		
		//rbm.trainTillConvergence(0.1, 1, input);
		
		
		ce = rbm.getReConstructionCrossEntropy();
		System.out.println("ce: " + ce);

		Matrix v = new DenseMatrix(new double[][]
				{
					{1, 1, 1, 0, 0, 0},
					{0, 0, 0, 1, 1, 1}
				}
		);	

		Matrix recon = rbm.reconstructVisibleInput(v);
		
		System.out.println("target vectors to reconstruct: ");
		MatrixUtils.debug_print(v);
		
		System.out.println("reconstruction: ");
		MatrixUtils.debug_print(recon);
		
		// "get the cross entropy somewhere near 0.3 and we're good"
		//assertEquals(0.5, ce, 0.2 );

		System.out.println("weights and biases: ");

		MatrixUtils.debug_print( rbm.connectionWeights );
		MatrixUtils.debug_print( rbm.visibleBiasNeurons );
		MatrixUtils.debug_print( rbm.hiddenBiasNeurons );
		

	}	
	
	@Test
	public void testRBMTrainingOnIrisDataset() throws IOException {
		
		Pair<Matrix, Matrix> data_set = IrisDatasetUtils.getIrisAsDataset();
		
		//MatrixUtils.debug_print(data_set.getFirst());
		//MatrixUtils.debug_print(data_set.getSecond());
		
		Matrix input = data_set.getFirst();
		//Matrix labels = data_set.getSecond();
		
		int visible_neuron_count = input.numCols();
		int hidden_neuron_count = visible_neuron_count / 2;
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(visible_neuron_count, hidden_neuron_count, null);


		
		double ce = 0;
		
		rbm.trainTillConvergence(0.001, 1, input);

		ce = rbm.getReConstructionCrossEntropy();
			
			
		ce = rbm.getReConstructionCrossEntropy();
		System.out.println("ce: " + ce);
		
	}
	
	

	/**
	 * Tests to see if the Cross Entropy drops below a certain level after 1000 epochs
	 * 
	 */
	@Test
	public void testCrossEntropyReconstructionOnSyntheticData() {
		
		//Matrix input = buildTestInputDataset();
		
		double[][] data_simple = new double[][]
				{
					{1,1,1,0,0,0},
					{0,0,0,1,1,1},
					{1,1,1,0,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,1,0}
					
				};
		
		Matrix input = new DenseMatrix(data_simple);		
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 4, null);
		rbm.useRegularization = false;
//		rbm.connectionWeights = rbm.connectionWeights.times( 100 );
		
		//MatrixUtils.debug_print( rbm.connectionWeights );
		//MatrixUtils.debug_print( rbm.visibleBiasNeurons );
		//MatrixUtils.debug_print( rbm.hiddenBiasNeurons );
		
		
		double ce = 0;
		
		for (int x = 0; x < 10; x++) {
			
			//rbm.contrastiveDivergence(0.01, 1, input);
			rbm.trainTillConvergence(0.01, 1, input);

			ce = rbm.getReConstructionCrossEntropy();
			
			
		
		}
		
		//rbm.trainTillConvergence(0.1, 1, input);
		
		
		ce = rbm.getReConstructionCrossEntropy();
		System.out.println("ce: " + ce);

		Matrix v = new DenseMatrix(new double[][]
				{
					{1, 1, 1, 0, 0, 0},
					{0, 0, 1, 1, 1, 0}
				}
		);	

		Matrix recon = rbm.reconstructVisibleInput(v);
		
		MatrixUtils.debug_print(v);
		MatrixUtils.debug_print(recon);
		
		// vector 0
		for ( int row = 0; row < v.numRows(); row++ ) {
		
			for ( int col = 0; col < v.numCols(); col++ ) {
			
				assertEquals( v.viewRow(row).get(col), recon.viewRow(row).get(col), 0.3 );
			
			}
			
		}
		
		
		// "get the cross entropy somewhere near 0.3 and we're good"
		//assertEquals(0.5, ce, 0.2 );
		
//		MatrixUtils.debug_print( rbm.connectionWeights );
//		MatrixUtils.debug_print( rbm.visibleBiasNeurons );
//		MatrixUtils.debug_print( rbm.hiddenBiasNeurons );
		
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
		rbm.useRegularization = false;
		double ce = 1000;
		
		//rbm.contrastiveDivergence(0.001, 10, input);

		for ( int x = 0; x < 10; x++) {
			rbm.trainTillConvergence(0.01, 1, input);
			ce = rbm.getReConstructionCrossEntropy();
		}
		
		Matrix recon = rbm.reconstruct( input );
		
		MatrixUtils.debug_print( input );
		MatrixUtils.debug_print( recon );
		
		
	}
	
	@Test
	public void testXORTrainTilConvergenceOptimizer() {
		
		//trainTillConvergence
		
		Matrix input = x_xor_Matrix;
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(2, 4, null);
		rbm.useRegularization = false;
		
		double ce = 0;
		


		rbm.trainTillConvergence(0.01, 1, input);

		ce = rbm.getReConstructionCrossEntropy();
		System.out.println("ce: " + ce);
		rbm.trainTillConvergence(0.01, 1, input);

		ce = rbm.getReConstructionCrossEntropy();
		System.out.println("ce: " + ce);
		
		ce = rbm.getReConstructionCrossEntropy();
		System.out.println("ce: " + ce);
		
		Matrix recon = rbm.reconstruct(x_xor_Matrix);
		
		MatrixUtils.debug_print(recon);
		
		

	}	
	
	
	@Test
	public void testSerdeMechanics() throws FileNotFoundException {
		
		String tmpFilename = "/tmp/RBMSerdeTest.model";
		
		Matrix input = buildTestInputDataset();
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null);
		
		rbm.trainTillConvergence(0.1, 1, input);
		
		// save / write the model
		
		FileOutputStream oFileOutStream = new FileOutputStream( tmpFilename, false);
		rbm.write( oFileOutStream );
		
		// read / load the model
		FileInputStream oFileInputStream = new FileInputStream( tmpFilename );
		
		RestrictedBoltzmannMachine rbm_deser = new RestrictedBoltzmannMachine( 1, 1, null ); 
		rbm_deser.load(oFileInputStream);
		
		assertEquals( rbm.numberHiddenNeurons, rbm_deser.numberHiddenNeurons );

		assertEquals( true, MatrixUtils.elementwiseSame(rbm.connectionWeights, rbm_deser.connectionWeights ) );
		assertEquals( true, MatrixUtils.elementwiseSame(rbm.hiddenBiasNeurons, rbm_deser.hiddenBiasNeurons ) );
		assertEquals( true, MatrixUtils.elementwiseSame(rbm.trainingDataset, rbm_deser.trainingDataset ) );
		assertEquals( true, MatrixUtils.elementwiseSame(rbm.visibleBiasNeurons, rbm_deser.visibleBiasNeurons ) );
		
		
	}	
	
	@Test
	public void testParameterAveragingSerdeMechanics() throws FileNotFoundException {
		
		String tmpFilename = "/tmp/RBMParamAvgSerdeTest.model";
		
		Matrix input = buildTestInputDataset();
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null);
		
		rbm.trainTillConvergence(0.1, 1, input);
		
		// save / write the model
		
		FileOutputStream oFileOutStream = new FileOutputStream( tmpFilename, false);
		rbm.serializeParameters( oFileOutStream );
		
		// read / load the model
		FileInputStream oFileInputStream = new FileInputStream( tmpFilename );
		
		RestrictedBoltzmannMachine rbm_deser = new RestrictedBoltzmannMachine( 1, 1, null ); 
		rbm_deser.loadParameterValues( oFileInputStream );
		
		assertEquals( rbm.numberHiddenNeurons, rbm_deser.numberHiddenNeurons );

		assertEquals( true, MatrixUtils.elementwiseSame(rbm.connectionWeights, rbm_deser.connectionWeights ) );
		assertEquals( true, MatrixUtils.elementwiseSame(rbm.hiddenBiasNeurons, rbm_deser.hiddenBiasNeurons ) );
		//assertEquals( true, MatrixUtils.elementwiseSame(rbm.trainingDataset, rbm_deser.trainingDataset ) );
		assertEquals( true, MatrixUtils.elementwiseSame(rbm.visibleBiasNeurons, rbm_deser.visibleBiasNeurons ) );
		
		
	}		
	
	
	
}
