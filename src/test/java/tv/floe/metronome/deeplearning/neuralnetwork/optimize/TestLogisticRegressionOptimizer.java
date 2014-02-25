package tv.floe.metronome.deeplearning.neuralnetwork.optimize;

import static org.junit.Assert.*;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.iterativereduce.iris.IrisDatasetUtils;
import tv.floe.metronome.datasets.UCIDatasets;
import tv.floe.metronome.deeplearning.neuralnetwork.core.LogisticRegression;
import tv.floe.metronome.deeplearning.neuralnetwork.core.LogisticRegressionGradient;
import tv.floe.metronome.deeplearning.neuralnetwork.optimize.util.CustomConjugateGradient;
import tv.floe.metronome.eval.Evaluation;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

public class TestLogisticRegressionOptimizer {


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
		
		//MatrixUtils.debug_print(x);
		
		x = gti(x, 0.5);
		
		//MatrixUtils.debug_print(x);
		
		
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
	
		//MatrixUtils.debug_print(y);
		
		Pair<Matrix, Matrix> p = new Pair<Matrix, Matrix>(x, y);
		
		return p;
		
	}	
	
	/**
	 * Mental Note: don't expect a linear classifier to get a perfect score on
	 * a non-linear modeling problem.
	 * 
	 * Madness, thy name is XOR
	 * 
	 */
	@Test
	public void testConjugateGradientLogisticRegressionOnXOR() {
		

		int n = 10;
		Pair<Matrix, Matrix> d = xorData(n);
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
		
		
		
		
		double learningRate = 0.01;
		
//		this.logisticRegressionLayer = new LogisticRegression(layer_input, this.hiddenLayerSizes[this.numberLayers-1], this.outputNeuronCount );
		
		
		LogisticRegression logreg = new LogisticRegression(x, 2, 2);
		
		logreg.labels = y;
		
		
		LogisticRegressionOptimizer opt = new LogisticRegressionOptimizer( logreg, learningRate );
		CustomConjugateGradient g = new CustomConjugateGradient(opt);
		g.optimize();
		
		
		Matrix predict = logreg.predict(x);
		//log.info(predict.toString());

		Evaluation eval = new Evaluation();
		eval.eval(y, predict);
		//log.info(eval.stats());
		System.out.println( eval.stats() );

		System.out.println( "Total Correct: " + eval.correctScores() + " out of " + n );

		
		
		
		
	}
	
	
	@Test
	public void testBasicConjugateGradientLogisticRegression() {
		
		double[][] x = new double[][] 
				{
						{1,1,1,0,0,0},
						{1,0,1,0,0,0},
						{1,1,1,0,0,0},
						{0,0,1,1,1,0},
						{0,0,1,1,0,0},
						{0,0,0,1,1,1}
						
				};
				
				double[][] y = new double[][] 
				{
						{1, 0},
						{1, 0},
						{1, 0},
						{0, 1},
						{0, 1},
						{0, 1}
				};
				
				Matrix xMatrix = new DenseMatrix(x);
				Matrix yMatrix = new DenseMatrix(y);

				double[][] xTest = new double[][] {
						{1, 1, 1, 0, 0, 0},
						{0, 0, 0, 1, 1, 1}
						//{1, 1, 1, 1, 1, 0}
				};
				
				Matrix xTestMatrix = new DenseMatrix(xTest);
				
					
				double learningRate = 0.001;
				
//				this.logisticRegressionLayer = new LogisticRegression(layer_input, this.hiddenLayerSizes[this.numberLayers-1], this.outputNeuronCount );
				
				
				LogisticRegression logreg = new LogisticRegression(xMatrix, 6, 2);
				
				logreg.labels = yMatrix;
				
				
				LogisticRegressionOptimizer opt = new LogisticRegressionOptimizer( logreg, learningRate );
				

				
				CustomConjugateGradient g = new CustomConjugateGradient(opt);
				g.optimize();
				
				
				Matrix predict = logreg.predict(xMatrix);
				//log.info(predict.toString());

				Evaluation eval = new Evaluation();
				eval.eval(yMatrix, predict);
				//log.info(eval.stats());
				System.out.println( eval.stats() );

				System.out.println( "Total Correct: " + eval.correctScores() + " out of " + yMatrix.numRows() );
				
				
		
		
	}
	
	@Test
	public void testTrainOnIrisDataset() throws IOException {
		
		double learningRate = 0.001;
		
		Pair<Matrix, Matrix> data_set = IrisDatasetUtils.getIrisAsDataset();
		
		MatrixUtils.debug_print(data_set.getFirst());
		MatrixUtils.debug_print(data_set.getSecond());
		
		Matrix input = data_set.getFirst();
		Matrix labels = data_set.getSecond();
		
		LogisticRegression logRegression = new LogisticRegression( input, input.numCols(), labels.numCols()); 
		logRegression.labels = labels;
		
		LogisticRegressionOptimizer opt = new LogisticRegressionOptimizer( logRegression, learningRate );
		CustomConjugateGradient g = new CustomConjugateGradient(opt);
		g.optimize();

		
		Matrix predict = logRegression.predict(input);
		
		MatrixUtils.debug_print(predict);

		Evaluation eval = new Evaluation();
		eval.eval(labels, predict);
		System.out.println( eval.stats() );

		System.out.println( "Total Correct: " + eval.correctScores() + " out of " + labels.numRows() );
		
		assertEquals( 0.95, eval.f1(), 0.02 );
		
		
	}	
	
	@Test
	public void testTrainOnCovTypeDataset() throws Exception {
		
		Pair<Matrix, Matrix> data_set = UCIDatasets.getCovTypeDataset(10000, 7); // 1-7, 0 class is empty
		
		//MatrixUtils.debug_print(data_set.getFirst());
		//MatrixUtils.debug_print(data_set.getSecond());
		
		Matrix input = data_set.getFirst();
		Matrix labels = data_set.getSecond();
		
		System.out.println( "Beginning LogReg Training on CovType");
		
		LogisticRegression logRegression = new LogisticRegression( input, input.numCols(), labels.numCols()); 

		double learningRate = 0.001;
		logRegression.labels = labels;
		
		LogisticRegressionOptimizer opt = new LogisticRegressionOptimizer( logRegression, learningRate );
		CustomConjugateGradient g = new CustomConjugateGradient(opt);
		g.optimize();
		
		
		
		Matrix predict = logRegression.predict(input);

		Evaluation eval = new Evaluation();
		eval.eval(labels, predict);
		//log.info(eval.stats());
		System.out.println( eval.stats() );

		System.out.println( "Total Correct: " + eval.correctScores() + " out of " + labels.numRows() );
		
		assertEquals( 0.95, eval.f1(), 0.1 );
		
		//MatrixUtils.debug_print(predict);		
		
	}		

}
