package tv.floe.metronome.deeplearning.neuralnetwork.dbn;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;

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
	
	public static Pair<Matrix, Matrix> xorDataStatic() {
		
		
		Matrix x = new DenseMatrix(4, 2); //Matrix.rand(n,2);
		Matrix y = new DenseMatrix( 4, 2 ); //Matrix.zeros(n,2);
		y.assign(0.0);

		
		/*
		MatrixUtils.debug_print(x);
		
		x = gti(x, 0.5);
		
		MatrixUtils.debug_print(x);
		
		
		
		for (int i = 0; i < x.numRows(); i++) {
			
			if (x.get(i,0) == x.get(i,1)) {
				
				y.set(i,0,1);
				
			} else {
				
				y.set(i,1,1);
				
			}
			
		}
		
		//return new DataSet(x,y);
	
		MatrixUtils.debug_print(y);
		*/
		
		/*
0 0:0 1:0
1 0:0 1:1
1 0:1 1:0
0 0:1 1:1

		 */
		
		x.viewRow(0).assign( new double[] { 0, 0 } );
		x.viewRow(1).assign( new double[] { 0, 1 } );
		x.viewRow(2).assign( new double[] { 1, 0 } );
		x.viewRow(3).assign( new double[] { 1, 1 } );
		
		y.set(0, 0, 1);
		y.set(1, 1, 1);
		y.set(2, 1, 1);
		y.set(3, 0, 1);
		
		Pair<Matrix, Matrix> p = new Pair<Matrix, Matrix>(x, y);
		
		MatrixUtils.debug_print(x);
		MatrixUtils.debug_print(y);
		
		return p;
		
	}
	
	
	@Test
	public void testXor() {

		//matrixRand(2, 2);
	//	xorData( 2 );
		
		int n = 10;
		Pair<Matrix, Matrix> d = xorDataStatic(); //xorData(n);
		Matrix x = d.getFirst();
		Matrix y = d.getSecond();





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

		
		
		dbn.preTrain(x,k, preTrainLr, preTrainEpochs);
		
		
		
		dbn.finetune(y,fineTuneLr, fineTuneEpochs);







		Matrix predict = dbn.predict(x);
		//log.info(predict.toString());

		Evaluation eval = new Evaluation();
		eval.eval(y, predict);
		//log.info(eval.stats());
		System.out.println( eval.stats() );

		System.out.println( "Total Correct: " + eval.correctScores() + " out of " + n );
	
	
	}
	
	
	@Test
	public void testSerdeMechanics() throws FileNotFoundException {
		
		String tmpFilename = "/tmp/DBNSerdeTest.model";
		
		
		int n = 10;
		Pair<Matrix, Matrix> d = xorDataStatic(); //xorData(n);
		Matrix x = d.getFirst();
		Matrix y = d.getSecond();





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

		
		
		dbn.preTrain(x,k, preTrainLr, preTrainEpochs);
		dbn.finetune(y,fineTuneLr, fineTuneEpochs);

		
		
		
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
		
		
		int n = 10;
		Pair<Matrix, Matrix> d = xorDataStatic(); //xorData(n);
		Matrix x = d.getFirst();
		Matrix y = d.getSecond();





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

		
		
		dbn.preTrain(x,k, preTrainLr, preTrainEpochs);
		dbn.finetune(y,fineTuneLr, fineTuneEpochs);

		
		
		
		// save / write the model
		
		FileOutputStream oFileOutStream = new FileOutputStream( tmpFilename, false);
		dbn.serializeParameters( oFileOutStream );
		
		// read / load the model
		FileInputStream oFileInputStream = new FileInputStream( tmpFilename );
		
		
		DeepBeliefNetwork dbn_deserialize = new DeepBeliefNetwork(1, hiddenLayerSizes, 1, hiddenLayerSizes.length, rng ); //, Matrix input, Matrix labels);
		dbn_deserialize.loadParameterValues( oFileInputStream );
		
		
		
		
		//assertEquals( dbn.inputNeuronCount, dbn_deserialize.inputNeuronCount );
		//assertEquals( dbn.l2, dbn_deserialize.l2, 0.0 );

		// check logistic layer
		
		assertEquals( true, MatrixUtils.elementwiseSame(dbn.logisticRegressionLayer.connectionWeights, dbn_deserialize.logisticRegressionLayer.connectionWeights ) );
		assertEquals( true, MatrixUtils.elementwiseSame(dbn.logisticRegressionLayer.biasTerms, dbn_deserialize.logisticRegressionLayer.biasTerms ) );

		
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
	
	
}
