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
	
	public static Pair<Matrix, Matrix> xorData(int n) {
		
		
		Matrix x = matrixRand(n, 2); //Matrix.rand(n,2);
		
		MatrixUtils.debug_print(x);
		
		x = gti(x, 0.5);
		
		MatrixUtils.debug_print(x);
		
		
		Matrix y = new DenseMatrix( n, 2 ); //Matrix.zeros(n,2);
		y.assign(0.0);
		
		for (int i = 0; i < x.numRows(); i++) {
			
			if (x.get(i,0) == x.get(i,1)) {
				
				y.set(i,0,1);
				
			} else {
				
				y.set(i,1,1);
				
			}
			
		}
		
		//return new DataSet(x,y);
	
		MatrixUtils.debug_print(y);
		
		Pair<Matrix, Matrix> p = new Pair<Matrix, Matrix>(x, y);
		
		return p;
		
	}
	
	
	@Test
	public void testXor() {

		//matrixRand(2, 2);
	//	xorData( 2 );
		
		int n = 10;
		Pair<Matrix, Matrix> d = xorData(n);
		Matrix x = d.getFirst();
		Matrix y = d.getSecond();





		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.001;
		int preTrainEpochs = 1000;
		int k = 1;
		int nIns = 2,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {2,2,2};
		double fineTuneLr = 0.001;
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

		
		dbn.preTrain(x,k, preTrainLr, preTrainEpochs);
		dbn.finetune(y,fineTuneLr, fineTuneEpochs);







		Matrix predict = dbn.predict(x);
		//log.info(predict.toString());

		Evaluation eval = new Evaluation();
		eval.eval(y, predict);
		//log.info(eval.stats());
		System.out.println( eval.stats() );

		
	
	
	}
	
	
	@Test
	public void testSerdeMechanics() throws FileNotFoundException {
		
		String tmpFilename = "/tmp/DBNSerdeTest.model";
		
		
		int n = 10;
		Pair<Matrix, Matrix> d = xorData(n);
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
		
		
		
/*		
		// read / load the model
		FileInputStream oFileInputStream = new FileInputStream( tmpFilename );
		
		LogisticRegression logRegression_deser = new LogisticRegression( null, 0, 0); 
		logRegression_deser.load(oFileInputStream);
		
		assertEquals( logRegression.nIn, logRegression_deser.nIn );
		assertEquals( logRegression.nOut, logRegression_deser.nOut );

		assertEquals( true, MatrixUtils.elementwiseSame(logRegression.input, logRegression_deser.input ) );
		assertEquals( true, MatrixUtils.elementwiseSame(logRegression.labels, logRegression_deser.labels ) );
		assertEquals( true, MatrixUtils.elementwiseSame(logRegression.connectionWeights, logRegression_deser.connectionWeights ) );
		assertEquals( true, MatrixUtils.elementwiseSame(logRegression.biasTerms, logRegression_deser.biasTerms ) );
		*/
		
	}		

}
