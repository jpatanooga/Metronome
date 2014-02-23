package tv.floe.metronome.deeplearning.neuralnetwork.optimize;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

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
	
	@Test
	public void testConjugateGradientLogisticRegressionOnXOR() {
		

		int n = 100;
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
		
		
		
		
		double learningRate = 0.001;
		
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
				
				// do asserts to make sure optimizer is working right
				
				assertEquals( 14, opt.getNumParameters() );
				double[] params = new double[14];
				opt.getParameters(params);
				
				// NLL
				//opt.getValue()
				
				//opt.getValueGradient(buffer)
				
				LogisticRegressionGradient grad = logreg.getGradient( learningRate );
				double[] buf = new double[ MatrixUtils.length( grad.getwGradient() ) + MatrixUtils.length( grad.getbGradient() ) ];
				
				System.out.println( "Weight Gradient Size: " + MatrixUtils.length( grad.getwGradient() ) );
				System.out.println( "Bias Gradient Size: " + MatrixUtils.length( grad.getbGradient() ) );
				
				opt.getValueGradient( buf );
				
				MatrixUtils.debug_print( grad.getwGradient() );
				MatrixUtils.debug_print( grad.getbGradient() );
				
				Matrix debug_buf = new DenseMatrix( 1, 14 );
				debug_buf.viewRow(0).assign(buf);
				
				MatrixUtils.debug_print( debug_buf );
				
				
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

}
