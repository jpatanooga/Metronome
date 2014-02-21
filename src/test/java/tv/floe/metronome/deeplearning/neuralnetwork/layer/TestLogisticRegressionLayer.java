package tv.floe.metronome.deeplearning.neuralnetwork.layer;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

import tv.floe.metronome.deeplearning.neuralnetwork.core.LogisticRegression;
import tv.floe.metronome.math.MatrixUtils;


public class TestLogisticRegressionLayer {

	double[][] x = new double[][] 
	{
			{1,1,1,0,0,0},
//			{1,0,1,0,0,0},
//			{1,1,1,0,0,0},
//			{0,0,1,1,1,0},
//			{0,0,1,1,0,0},
			{0,0,0,1,1,1}
			
	};
	
	double[][] y = new double[][] 
	{
			{1, 0},
//			{1, 0},
//			{1, 0},
//			{0, 1},
//			{0, 1},
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
	
	
	
	/**
	 * Need to make sure the math on the train method's linear algebra is legit
	 * 
	 */
	@Test
	public void testTrainMethod() {
		//fail("Not yet implemented");
		
/*
		LogisticRegression log2 = new LogisticRegression(xTestMatrix,x[0].length,2);
		double learningRate = 0.01;
		for(int i = 0; i < 1000; i++) {
			log2.train(xMatrix, yMatrix, learningRate);
			learningRate *= 0.95;
		}
		
 */
		
		LogisticRegression logRegression = new LogisticRegression( xTestMatrix, x[0].length, 2); 

		double learningRate = 0.001;
		
		for (int i = 0; i < 10000; i++) {
			
			logRegression.train(xMatrix, yMatrix, learningRate);
			learningRate *= 0.999;

		}
		
		Matrix predictions = logRegression.predict(xTestMatrix);
		
		MatrixUtils.debug_print(predictions);
		
	}
	
	@Test
	public void testAdagradTrainMethod() {
		
		LogisticRegression logRegression = new LogisticRegression( xTestMatrix, x[0].length, 2); 

		double learningRate = 0.001;
		
		for (int i = 0; i < 10000; i++) {
			
			//logRegression.train(xMatrix, yMatrix, learningRate);
			logRegression.trainWithAdagrad(xMatrix, yMatrix);

		}
		
		Matrix predictions = logRegression.predict(xTestMatrix);
		
		MatrixUtils.debug_print(predictions);
		
	}	
	

}
