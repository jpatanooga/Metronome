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

import tv.floe.metronome.math.MatrixUtils;


public class TestLogisticRegressionLayer {

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
		
		LogisticRegressionLayer logRegression = new LogisticRegressionLayer( xTestMatrix, x[0].length, 2); 

		double learningRate = 0.01;
		
		for (int i = 0; i < 1000; i++) {
			
			logRegression.train(xMatrix, yMatrix, learningRate);
			learningRate *= 0.95;

		}
		
		Matrix predictions = logRegression.predict(xTestMatrix);
		
		MatrixUtils.debug_print(predictions);
		
	}
	

	@Test
	public void testSerdeMechanics() throws FileNotFoundException {
		
		String tmpFilename = "/tmp/logisticLayerTest.model";
		
		LogisticRegressionLayer logRegression = new LogisticRegressionLayer( xTestMatrix, x[0].length, 2); 

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
		
		LogisticRegressionLayer logRegression_deser = new LogisticRegressionLayer( null, 0, 0); 
		logRegression_deser.load(oFileInputStream);
		
		assertEquals( logRegression.numInputNeurons, logRegression_deser.numInputNeurons );
		assertEquals( logRegression.numOutputNeurons, logRegression_deser.numOutputNeurons );
		
		Matrix a = new DenseMatrix( 1, 2 );
		a.set(0, 0, 1 );
		a.set(0, 1, 2 );
		
		Matrix b = new DenseMatrix( 1, 2 );
		b.set(0, 0, 1 );
		b.set(0, 1, 2 );
		
		System.out.println( "matrix equals: " + a.equals(b) );
		
		/*
		
		for ( int x = 0; x < layer_deser.input.numRows(); x++) {
			
			for ( int c = 0; c < layer_deser.input.numCols(); c++ ) {
			
				//layer_deser.input.viewRow(x)
				assertEquals( input.get(x, c), layer_deser.input.get(x, c), 0.0 );
				
			}
			
		}
		
		*/
		
		
	}	

}
