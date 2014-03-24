package tv.floe.metronome.deeplearning.neuralnetwork.dbn;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;


import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;
import tv.floe.metronome.deeplearning.math.transforms.MultiplyScalar;
import tv.floe.metronome.deeplearning.neuralnetwork.core.LogisticRegression;
import tv.floe.metronome.eval.Evaluation;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;


public class TestDeepBeliefNetwork {


	double[][] x_toy = new double[][] 
	{
			{1,1,1,0,0,0},
			{1,0,1,0,0,0},
			{1,1,1,0,0,0},
			{0,0,1,1,1,0},
			{0,0,1,1,0,0},
			{0,0,0,1,1,1}
			
	};
	
	double[][] y_toy = new double[][] 
	{
			{1, 0},
			{1, 0},
			{1, 0},
			{0, 1},
			{0, 1},
			{0, 1}
	};
	
	Matrix x_toy_Matrix = new DenseMatrix(x_toy);
	Matrix y_toy_Matrix = new DenseMatrix(y_toy);

	double[][] x_toy_Test = new double[][] {
			{1, 1, 1, 0, 0, 0},
			{0, 0, 0, 1, 1, 1}
			//{1, 1, 1, 1, 1, 0}
	};
	
	Matrix xTestMatrix = new DenseMatrix(x_toy_Test);
		
	
	
	public static Matrix gti(Matrix m, double val) {
		
		Matrix res = m.like();
		
		for ( int r = 0; r < m.numRows(); r++ ) {
			
			for ( int c = 0; c < m.numCols(); c++ ) {
				
				if (m.get(r, c) > val) {
					res.set(r, c, 1.0);
				} else {
					res.set(r, c, 0.0);
				}
				
			}
			
			
		}
		
		return res;
		
	}
	
	public static Matrix matrixRand( int rows, int cols ) {
		
		RandomGenerator rnd = new MersenneTwister(1234);
		//rnd.nextDouble();
		
		Matrix res = new DenseMatrix( rows, cols );
		
		for ( int r = 0; r < res.numRows(); r++ ) {
			
			for ( int c = 0; c < res.numCols(); c++ ) {
				
				res.set(r,  c, rnd.nextDouble());
				//System.out.println( "next: " + rnd.nextDouble() );
				
			}
			
		}
		
		return res;
		
	}

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
	Matrix y_xor_Matrix = new DenseMatrix(xor_labels);

	
	
	@Test
	public void testXor() {






		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.0001;
		int preTrainEpochs = 1000;
		int k = 1;
		int nIns = 2,nOuts = 2;
		int[] hiddenLayerSizes = new int[] { 2 };
		double fineTuneLr = 0.0001;
		int fineTuneEpochs = 1000;
/*
		DBN dbn = new DBN.Builder()
		.transformWeightsAt(0, new MultiplyScalar(1000))
		.transformWeightsAt(1, new MultiplyScalar(100))

		.hiddenLayerSizes(hiddenLayerSizes).numberOfInputs(nIns).renderWeights(0)
		.useRegularization(false).withMomentum(0).withDist(new NormalDistribution(0,0.001))
		.numberOfOutPuts(nOuts).withRng(rng).build();
*/
		
		DeepBeliefNetwork dbn = new DeepBeliefNetwork(nIns, hiddenLayerSizes, nOuts, hiddenLayerSizes.length, rng ); //, Matrix input, Matrix labels);
//		dbn.addWeightTransform(0, new MultiplyScalar(100));
//		dbn.addWeightTransform(1, new MultiplyScalar(10));

		
		
		dbn.preTrain(x_xor_Matrix,k, preTrainLr, preTrainEpochs);
		
		
		
		dbn.finetune(y_xor_Matrix,fineTuneLr, fineTuneEpochs);







		Matrix predict = dbn.predict( x_xor_Matrix );
		System.out.println("--- Predictions XOR ----");
		MatrixUtils.debug_print(predict);
		//log.info(predict.toString());

		Evaluation eval = new Evaluation();
		eval.eval( y_xor_Matrix, predict );
		//log.info(eval.stats());
		System.out.println( eval.stats() );

		System.out.println( "Total Correct: " + eval.correctScores() + " out of " + x_xor_Matrix.numRows() );
	
	
	}
	

	
	@Test
	public void testToyInput() {






		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.0001;
		int preTrainEpochs = 1000;
		int k = 1;
		
		int nIns = x_toy[0].length;
		int nOuts = y_toy[0].length;
		
		int[] hiddenLayerSizes = new int[] { 10, 8 };
		double fineTuneLr = 0.0001;
		int fineTuneEpochs = 1000;
		
		MatrixUtils.debug_print(x_toy_Matrix);
		
/*
		DBN dbn = new DBN.Builder()
		.transformWeightsAt(0, new MultiplyScalar(1000))
		.transformWeightsAt(1, new MultiplyScalar(100))

		.hiddenLayerSizes(hiddenLayerSizes).numberOfInputs(nIns).renderWeights(0)
		.useRegularization(false).withMomentum(0).withDist(new NormalDistribution(0,0.001))
		.numberOfOutPuts(nOuts).withRng(rng).build();
*/
		
		DeepBeliefNetwork dbn = new DeepBeliefNetwork(nIns, hiddenLayerSizes, nOuts, hiddenLayerSizes.length, rng ); //, Matrix input, Matrix labels);
//		dbn.addWeightTransform(0, new MultiplyScalar(100));
//		dbn.addWeightTransform(1, new MultiplyScalar(10));

		
		
		dbn.preTrain(x_toy_Matrix,k, preTrainLr, preTrainEpochs);
		
		
		
		dbn.finetune(y_toy_Matrix,fineTuneLr, fineTuneEpochs);







		Matrix predict = dbn.predict( x_toy_Matrix );
		System.out.println("--- Predictions Toy Matrix ----");
		MatrixUtils.debug_print(predict);
		//log.info(predict.toString());

		Evaluation eval = new Evaluation();
		eval.eval( y_toy_Matrix, predict );
		//log.info(eval.stats());
		System.out.println( eval.stats() );

		System.out.println( "Total Correct: " + eval.correctScores() + " out of " + x_toy_Matrix.numRows() );
	
	
	}	
	
	
	@Test
	public void testSerdeMechanics() throws FileNotFoundException {
		
		String tmpFilename = "/tmp/DBNSerdeTest.model";
		
		/*
		int n = 10;
		Pair<Matrix, Matrix> d = xorDataStatic(); //xorData(n);
		Matrix x = d.getFirst();
		Matrix y = d.getSecond();
*/




		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.001;
		int preTrainEpochs = 100;
		int k = 1;
		int nIns = 2,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {2,2,2};
		double fineTuneLr = 0.001;
		int fineTuneEpochs = 100;
/*
		DBN dbn = new DBN.Builder()
		.transformWeightsAt(0, new MultiplyScalar(1000))
		.transformWeightsAt(1, new MultiplyScalar(100))

		.hiddenLayerSizes(hiddenLayerSizes).numberOfInputs(nIns).renderWeights(0)
		.useRegularization(false).withMomentum(0).withDist(new NormalDistribution(0,0.001))
		.numberOfOutPuts(nOuts).withRng(rng).build();
*/
		
		DeepBeliefNetwork dbn = new DeepBeliefNetwork(nIns, hiddenLayerSizes, nOuts, hiddenLayerSizes.length, rng ); //, Matrix input, Matrix labels);

		
		
		dbn.preTrain(x_xor_Matrix,k, preTrainLr, preTrainEpochs);
		dbn.finetune(y_xor_Matrix,fineTuneLr, fineTuneEpochs);

		
		
		
		// save / write the model
		
		FileOutputStream oFileOutStream = new FileOutputStream( tmpFilename, false);
		dbn.write( oFileOutStream );
		
		// read / load the model
		FileInputStream oFileInputStream = new FileInputStream( tmpFilename );
		
		
		DeepBeliefNetwork dbn_deserialize = new DeepBeliefNetwork(1, hiddenLayerSizes, 1, hiddenLayerSizes.length, rng ); //, Matrix input, Matrix labels);
		dbn_deserialize.load(oFileInputStream);
		
		
		
		
		assertEquals( dbn.inputNeuronCount, dbn_deserialize.inputNeuronCount );
		assertEquals( dbn.l2, dbn_deserialize.l2, 0.0 );

		// check hidden layers
		for ( int i = 0; i < dbn.hiddenLayers.length; i++ ) {
			
			assertEquals( true, MatrixUtils.elementwiseSame(dbn.hiddenLayers[i].biasTerms, dbn_deserialize.hiddenLayers[i].biasTerms ) );
			assertEquals( true, MatrixUtils.elementwiseSame(dbn.hiddenLayers[i].connectionWeights, dbn_deserialize.hiddenLayers[i].connectionWeights ) );

		}
		
		// check RBMs
		for ( int i = 0; i < dbn.preTrainingLayers.length; i++ ) {
			
			assertEquals( true, MatrixUtils.elementwiseSame(dbn.preTrainingLayers[i].getConnectionWeights(), dbn_deserialize.preTrainingLayers[i].getConnectionWeights() ) );
			assertEquals( true, MatrixUtils.elementwiseSame(dbn.preTrainingLayers[i].getHiddenBias(), dbn_deserialize.preTrainingLayers[i].getHiddenBias() ) );
			assertEquals( true, MatrixUtils.elementwiseSame(dbn.preTrainingLayers[i].getVisibleBias(), dbn_deserialize.preTrainingLayers[i].getVisibleBias() ) );

		}
		
	}		

	
	@Test
	public void testParameterAvgSerdeMechanics() throws FileNotFoundException {
		
		String tmpFilename = "/tmp/DBNParameterAvgSerdeTest.model";
		
		



		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.001;
		int preTrainEpochs = 100;
		int k = 1;
		int nIns = 2,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {2,2,2};
		double fineTuneLr = 0.001;
		int fineTuneEpochs = 100;
/*
		DBN dbn = new DBN.Builder()
		.transformWeightsAt(0, new MultiplyScalar(1000))
		.transformWeightsAt(1, new MultiplyScalar(100))

		.hiddenLayerSizes(hiddenLayerSizes).numberOfInputs(nIns).renderWeights(0)
		.useRegularization(false).withMomentum(0).withDist(new NormalDistribution(0,0.001))
		.numberOfOutPuts(nOuts).withRng(rng).build();
*/
		
		DeepBeliefNetwork dbn = new DeepBeliefNetwork(nIns, hiddenLayerSizes, nOuts, hiddenLayerSizes.length, rng ); //, Matrix input, Matrix labels);

		
		
		dbn.preTrain( x_xor_Matrix,k, preTrainLr, preTrainEpochs );
		dbn.finetune( y_xor_Matrix,fineTuneLr, fineTuneEpochs );

		
		
		
		// save / write the model
		
		FileOutputStream oFileOutStream = new FileOutputStream( tmpFilename, false);
		//dbn.serializeParameters( oFileOutStream );
		dbn.write(oFileOutStream);
		
		// read / load the model
		FileInputStream oFileInputStream = new FileInputStream( tmpFilename );
		
		
		DeepBeliefNetwork dbn_deserialize = new DeepBeliefNetwork(1, hiddenLayerSizes, 1, hiddenLayerSizes.length, rng ); //, Matrix input, Matrix labels);
		//dbn_deserialize.loadParameterValues( oFileInputStream );
		dbn_deserialize.load(oFileInputStream);
		
		int[] hiddenLayerSizesTmp = new int[] {1};
		
		// now setup a DBN based on a clone operation via initBasedOn()
		DeepBeliefNetwork dbn_merge_load = new DeepBeliefNetwork(1, hiddenLayerSizesTmp, 1, hiddenLayerSizesTmp.length, null); //1, , 1, hiddenLayerSizes.length, rng );
		dbn_merge_load.initBasedOn(dbn_deserialize);
		
		DeepBeliefNetwork dbn_merge_load_2 = new DeepBeliefNetwork(1, hiddenLayerSizesTmp, 1, hiddenLayerSizesTmp.length, null); //1, , 1, hiddenLayerSizes.length, rng );
		dbn_merge_load_2.initBasedOn( dbn );
		
		//assertEquals( dbn.inputNeuronCount, dbn_deserialize.inputNeuronCount );
		//assertEquals( dbn.l2, dbn_deserialize.l2, 0.0 );

		// check logistic layer
		
		assertEquals( true, MatrixUtils.elementwiseSame(dbn.logisticRegressionLayer.connectionWeights, dbn_deserialize.logisticRegressionLayer.connectionWeights ) );
		assertEquals( true, MatrixUtils.elementwiseSame(dbn.logisticRegressionLayer.connectionWeights, dbn_merge_load.logisticRegressionLayer.connectionWeights ) );
		
		assertEquals( true, MatrixUtils.elementwiseSame(dbn.logisticRegressionLayer.biasTerms, dbn_deserialize.logisticRegressionLayer.biasTerms ) );
		assertEquals( true, MatrixUtils.elementwiseSame(dbn.logisticRegressionLayer.biasTerms, dbn_merge_load.logisticRegressionLayer.biasTerms ) );

		
		// check hidden layers
		for ( int i = 0; i < dbn.hiddenLayers.length; i++ ) {
			
			assertEquals( true, MatrixUtils.elementwiseSame(dbn.hiddenLayers[i].biasTerms, dbn_deserialize.hiddenLayers[i].biasTerms ) );
			assertEquals( true, MatrixUtils.elementwiseSame(dbn.hiddenLayers[i].connectionWeights, dbn_deserialize.hiddenLayers[i].connectionWeights ) );

			assertEquals( dbn.hiddenLayers[i].biasTerms.numCols(), dbn_merge_load.hiddenLayers[i].biasTerms.numCols() );
			assertEquals( dbn.hiddenLayers[i].connectionWeights.numCols(), dbn_merge_load.hiddenLayers[i].connectionWeights.numCols() );
			
			
		}
		
		// check RBMs
		for ( int i = 0; i < dbn.preTrainingLayers.length; i++ ) {
			
			assertEquals( true, MatrixUtils.elementwiseSame(dbn.preTrainingLayers[i].getConnectionWeights(), dbn_deserialize.preTrainingLayers[i].getConnectionWeights() ) );
			assertEquals( true, MatrixUtils.elementwiseSame(dbn.preTrainingLayers[i].getHiddenBias(), dbn_deserialize.preTrainingLayers[i].getHiddenBias() ) );
			assertEquals( true, MatrixUtils.elementwiseSame(dbn.preTrainingLayers[i].getVisibleBias(), dbn_deserialize.preTrainingLayers[i].getVisibleBias() ) );

			assertEquals( dbn.preTrainingLayers[i].getConnectionWeights().numCols(), dbn_merge_load.preTrainingLayers[i].getConnectionWeights().numCols() );
			
			for ( int row = 0; row < dbn.preTrainingLayers[i].getConnectionWeights().numRows(); row++ ) {

				for ( int col = 0; col < dbn.preTrainingLayers[i].getConnectionWeights().numCols(); col++ ) {
				
					assertEquals( 0.0, dbn_merge_load.preTrainingLayers[i].getConnectionWeights().get( row, col), 0.0 );
					
				}
				
			}
			
			assertEquals( dbn.preTrainingLayers[i].getHiddenBias().numCols(), dbn_merge_load.preTrainingLayers[i].getHiddenBias().numCols() );

			for ( int row = 0; row < dbn.preTrainingLayers[i].getHiddenBias().numRows(); row++ ) {

				for ( int col = 0; col < dbn.preTrainingLayers[i].getHiddenBias().numCols(); col++ ) {

					assertEquals( 0.0, dbn_merge_load.preTrainingLayers[i].getHiddenBias().get( row, col), 0.0 );
					
				}
				
			}
			
			assertEquals( dbn.preTrainingLayers[i].getVisibleBias().numCols(), dbn_merge_load.preTrainingLayers[i].getVisibleBias().numCols() );
			
			for ( int row = 0; row < dbn.preTrainingLayers[i].getVisibleBias().numRows(); row++ ) {

				for ( int col = 0; col < dbn.preTrainingLayers[i].getVisibleBias().numCols(); col++ ) {

					assertEquals( 0.0, dbn_merge_load.preTrainingLayers[i].getVisibleBias().get( row, col), 0.0 );
					
				}
				
			}
			
		}
		
	}	
	
	
	@Test
	public void testParameterAveragingCode() {
		

		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.001;
		int preTrainEpochs = 100;
		int k = 1;
		int nIns = 2,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {2,2};
		double fineTuneLr = 0.001;
		int fineTuneEpochs = 100;
		
		DeepBeliefNetwork dbn_A = new DeepBeliefNetwork(nIns, hiddenLayerSizes, nOuts, hiddenLayerSizes.length, rng, x_xor_Matrix, y_xor_Matrix );
		
		// setup layer 0
		dbn_A.preTrainingLayers[ 0 ].getConnectionWeights().set(0, 0, 1.0);
		dbn_A.preTrainingLayers[ 0 ].getConnectionWeights().set(0, 1, 1.0);
		dbn_A.preTrainingLayers[ 0 ].getConnectionWeights().set(1, 0, 1.0);
		dbn_A.preTrainingLayers[ 0 ].getConnectionWeights().set(1, 1, 1.0);
		
		// Layer 0: hidden bias
		
		dbn_A.preTrainingLayers[ 0 ].getHiddenBias().set( 0, 0, 15.0 );
		dbn_A.preTrainingLayers[ 0 ].getHiddenBias().set( 0, 1, 16.0 );
				
		// Layer 0: visible bias

		dbn_A.preTrainingLayers[ 0 ].getVisibleBias().set( 0, 0, 25.0 );
		dbn_A.preTrainingLayers[ 0 ].getVisibleBias().set( 0, 1, 26.0 );
		
		
		// setup layer 1
		dbn_A.preTrainingLayers[ 1 ].getConnectionWeights().set(0, 0, 1.0);
		dbn_A.preTrainingLayers[ 1 ].getConnectionWeights().set(0, 1, 1.0);
		dbn_A.preTrainingLayers[ 1 ].getConnectionWeights().set(1, 0, 1.0);
		dbn_A.preTrainingLayers[ 1 ].getConnectionWeights().set(1, 1, 1.0);

		
		
		// setup log layer
		dbn_A.logisticRegressionLayer.connectionWeights.set(0, 0, 1.0);
		dbn_A.logisticRegressionLayer.connectionWeights.set(0, 1, 1.0);
		dbn_A.logisticRegressionLayer.connectionWeights.set(1, 0, 1.0);
		dbn_A.logisticRegressionLayer.connectionWeights.set(1, 1, 1.0);
		
		MatrixUtils.debug_print( dbn_A.logisticRegressionLayer.biasTerms );

		// set up bias terms
		
		dbn_A.logisticRegressionLayer.biasTerms.set( 0, 0, 35.0 );
		dbn_A.logisticRegressionLayer.biasTerms.set( 0, 1, 36.0 );
		
		
		
		
		DeepBeliefNetwork dbn_B = new DeepBeliefNetwork(nIns, hiddenLayerSizes, nOuts, hiddenLayerSizes.length, rng, x_xor_Matrix, y_xor_Matrix );

		dbn_B.preTrainingLayers[ 0 ].getConnectionWeights().set(0, 0, 2.0);
		dbn_B.preTrainingLayers[ 0 ].getConnectionWeights().set(0, 1, 1.0);
		dbn_B.preTrainingLayers[ 0 ].getConnectionWeights().set(1, 0, 3.0);
		dbn_B.preTrainingLayers[ 0 ].getConnectionWeights().set(1, 1, 4.0);

		// Layer 0: hidden bias
		
		dbn_B.preTrainingLayers[ 0 ].getHiddenBias().set( 0, 0, 5.0 );
		dbn_B.preTrainingLayers[ 0 ].getHiddenBias().set( 0, 1, 6.0 );
				
		// Layer 0: visible bias

		dbn_B.preTrainingLayers[ 0 ].getVisibleBias().set( 0, 0, 5.0 );
		dbn_B.preTrainingLayers[ 0 ].getVisibleBias().set( 0, 1, 6.0 );

		// layer 2
		
		dbn_B.preTrainingLayers[ 1 ].getConnectionWeights().set(0, 0, 2.0);
		dbn_B.preTrainingLayers[ 1 ].getConnectionWeights().set(0, 1, 1.0);
		dbn_B.preTrainingLayers[ 1 ].getConnectionWeights().set(1, 0, 3.0);
		dbn_B.preTrainingLayers[ 1 ].getConnectionWeights().set(1, 1, 4.0);
		
		dbn_B.logisticRegressionLayer.connectionWeights.set(0, 0, 1.0);
		dbn_B.logisticRegressionLayer.connectionWeights.set(0, 1, 2.0);
		dbn_B.logisticRegressionLayer.connectionWeights.set(1, 0, 3.0);
		dbn_B.logisticRegressionLayer.connectionWeights.set(1, 1, 4.0);
		
		dbn_B.logisticRegressionLayer.biasTerms.set( 0, 0, 5.0 );
		dbn_B.logisticRegressionLayer.biasTerms.set( 0, 1, 6.0 );
		
		
		// MatrixUtils.debug_print( dbn_B.preTrainingLayers[ 0 ].getConnectionWeights() );
		
		int[] hiddenLayerSizesTmp = new int[] {1};
		
		// now setup a DBN based on a clone operation via initBasedOn()
//		DeepBeliefNetwork dbn_merge_load = new DeepBeliefNetwork(1, hiddenLayerSizesTmp, 1, hiddenLayerSizesTmp.length, null);
		
		// this only works if the workers have been initialized on data
		DeepBeliefNetwork dbn_master = new DeepBeliefNetwork(1, hiddenLayerSizesTmp, 1, hiddenLayerSizesTmp.length, null);
		dbn_master.initBasedOn(dbn_A);
		
		
		ArrayList<DeepBeliefNetwork> workers = new ArrayList<DeepBeliefNetwork>();
		workers.add(dbn_A);
		workers.add(dbn_B);
		
		dbn_master.computeAverageDBNParameterVector(workers);
		
		assertEquals( 1.5, dbn_master.preTrainingLayers[ 0 ].getConnectionWeights().get(0, 0), 0.0 );
		assertEquals( 1.0, dbn_master.preTrainingLayers[ 0 ].getConnectionWeights().get(0, 1), 0.0 );
		assertEquals( 2.0, dbn_master.preTrainingLayers[ 0 ].getConnectionWeights().get(1, 0), 0.0 );
		assertEquals( 2.5, dbn_master.preTrainingLayers[ 0 ].getConnectionWeights().get(1, 1), 0.0 );

		// check hidden bias averaged weights
		
		assertEquals( 10.0, dbn_master.preTrainingLayers[ 0 ].getHiddenBias().get(0, 0), 0.0 );
		assertEquals( 11.0, dbn_master.preTrainingLayers[ 0 ].getHiddenBias().get(0, 1), 0.0 );

		// check visible bias averaged weights
		
		assertEquals( 15.0, dbn_master.preTrainingLayers[ 0 ].getVisibleBias().get(0, 0), 0.0 );
		assertEquals( 16.0, dbn_master.preTrainingLayers[ 0 ].getVisibleBias().get(0, 1), 0.0 );
		
		// now make sure the stock hidden weights are the same as the pre-train layer
		
		assertEquals( true, MatrixUtils.elementwiseSame( dbn_master.preTrainingLayers[ 0 ].getConnectionWeights(), dbn_master.hiddenLayers[ 0 ].connectionWeights ) );
		// hidden bias == bias ?
		assertEquals( true, MatrixUtils.elementwiseSame( dbn_master.preTrainingLayers[ 0 ].getHiddenBias(), dbn_master.hiddenLayers[ 0 ].biasTerms ) );

		
		assertEquals( 1.5, dbn_master.preTrainingLayers[ 1 ].getConnectionWeights().get(0, 0), 0.0 );
		assertEquals( 1.0, dbn_master.preTrainingLayers[ 1 ].getConnectionWeights().get(0, 1), 0.0 );
		assertEquals( 2.0, dbn_master.preTrainingLayers[ 1 ].getConnectionWeights().get(1, 0), 0.0 );
		assertEquals( 2.5, dbn_master.preTrainingLayers[ 1 ].getConnectionWeights().get(1, 1), 0.0 );
		
		
		assertEquals( 1.0, dbn_master.logisticRegressionLayer.connectionWeights.get(0, 0), 0.0 );		
		assertEquals( 1.5, dbn_master.logisticRegressionLayer.connectionWeights.get(0, 1), 0.0 );
		assertEquals( 2.0, dbn_master.logisticRegressionLayer.connectionWeights.get(1, 0), 0.0 );
		assertEquals( 2.5, dbn_master.logisticRegressionLayer.connectionWeights.get(1, 1), 0.0 );

		assertEquals( 20.0, dbn_master.logisticRegressionLayer.biasTerms.get(0, 0), 0.0 );
		assertEquals( 21.0, dbn_master.logisticRegressionLayer.biasTerms.get(0, 1), 0.0 );

		// TODO: check the hidden layers ---- 
		
		
		
	}
	
	
}
