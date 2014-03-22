package tv.floe.metronome.deeplearning.neuralnetwork.core;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

import tv.floe.metronome.deeplearning.neuralnetwork.layer.LogisticRegressionLayer;
import tv.floe.metronome.math.MatrixUtils;

public class TestDBNLogisticRegression {


	double[][] x = new double[][] 
	{
			{1,1,1,0,0,0},
			{1,0,1,0,0,0},
			{1,1,1,0,0,0},
			{0,0,1,1,1,0},
			{0,0,1,1,0,0},
			{0,0,1,1,1,0}
			
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
			{0, 0, 1, 1, 1, 0}
			//{1, 1, 1, 1, 1, 0}
	};
	
	Matrix xTestMatrix = new DenseMatrix(xTest);
		
	
	@Test
	public void testSerdeMechanics() throws FileNotFoundException {
		
		String tmpFilename = "/tmp/logisticRegressionTest.model";
		
		LogisticRegression logRegression = new LogisticRegression( xTestMatrix, x[0].length, 2); 

		double learningRate = 0.01;
		
		for (int i = 0; i < 1000; i++) {
			
			logRegression.train(xMatrix, yMatrix, learningRate);
			learningRate *= 0.95;

		}		
		
		
		
		// save / write the model
		
		FileOutputStream oFileOutStream = new FileOutputStream( tmpFilename, false);
		logRegression.write( oFileOutStream );
		
		
		
		
		// read / load the model
		FileInputStream oFileInputStream = new FileInputStream( tmpFilename );
		
		LogisticRegression logRegression_deser = new LogisticRegression( null, 0, 0); 
		logRegression_deser.load(oFileInputStream);
		
		assertEquals( logRegression.nIn, logRegression_deser.nIn );
		assertEquals( logRegression.nOut, logRegression_deser.nOut );

//		assertEquals( true, MatrixUtils.elementwiseSame(logRegression.input, logRegression_deser.input ) );
//		assertEquals( true, MatrixUtils.elementwiseSame(logRegression.labels, logRegression_deser.labels ) );
		assertEquals( true, MatrixUtils.elementwiseSame(logRegression.connectionWeights, logRegression_deser.connectionWeights ) );
		assertEquals( true, MatrixUtils.elementwiseSame(logRegression.biasTerms, logRegression_deser.biasTerms ) );
		
		
	}	
}
