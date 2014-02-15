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


public class TestHiddenLayer {

	Matrix input = new DenseMatrix(new double[][] 
	{
			{1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
			,{1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
			,{1,1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0}
			,{1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
			,{0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0}
			,{0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1}
			,{0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1}
			,{0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,0,1}
			,{0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1}
			,{0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0}
    }
	);	
	
	@Test
	public void testSimple() {

		
		MatrixUtils.debug_print_matrix_stats( input, "input" );
		
		
		
		
		
		RandomGenerator r = new MersenneTwister(1234);
		
		HiddenLayer layer = new HiddenLayer(20, 2, r);
		layer.setInput(input);
		
		MatrixUtils.debug_print_matrix_stats( layer.connectionWeights, "connection" );
		
		MatrixUtils.debug_print_matrix_stats( layer.sampleHiddenGivenLastVisible(), "last visible" );
		
		MatrixUtils.debug_print_matrix_stats( layer.computeActivationOutput(), "activation output" );
		
		assertEquals( 10, layer.sampleHiddenGivenLastVisible().numRows() );
		assertEquals( 2, layer.sampleHiddenGivenLastVisible().numCols() );
		
		assertEquals( 10, layer.computeActivationOutput().numRows() );
		assertEquals( 2, layer.computeActivationOutput().numCols() );
		
	}
	
	/**
	 * TODO: complete
	 * 
	 */
	@Test
	public void testComputeActivationOutput() {
		
		Matrix input = new DenseMatrix(new double[][] 
		{
				{1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
				,{1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
				,{1,1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0}
				,{1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
				,{0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0}
				,{0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1}
				,{0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1}
				,{0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,0,1}
				,{0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1}
				,{0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0}
        }
		);

		RandomGenerator r = new MersenneTwister(1234);
		
		HiddenLayer layer = new HiddenLayer(20, 2, r);
		layer.setInput(input);	
		
		Matrix output = layer.computeActivationOutput();
		
	}

	/**
	 * TODO: complete
	 * 
	 */	
	@Test 
	public void testSampleHiddenGivenLastVisible() {
		
	}
	

	@Test
	public void testSerdeMechanics() throws FileNotFoundException {
		
		String tmpFilename = "/tmp/hiddenLayerTest.model";
		
		HiddenLayer layer = new HiddenLayer(20, 2, null);
		layer.setInput(input);
		
		
		
		// save / write the model
		
		FileOutputStream oFileOutStream = new FileOutputStream( tmpFilename, false);
		layer.write( oFileOutStream );
		
		
		
		
		// read / load the model
		FileInputStream oFileInputStream = new FileInputStream( tmpFilename );
		
		HiddenLayer layer_deser = new HiddenLayer( 1, 1, null); 
		layer_deser.load(oFileInputStream);
		
		assertEquals( layer.neuronCount, layer_deser.neuronCount );
		assertEquals( layer.neuronCountPreviousLayer, layer_deser.neuronCountPreviousLayer );
		

		assertEquals( true, MatrixUtils.elementwiseSame(layer.biasTerms, layer_deser.biasTerms ) );
		assertEquals( true, MatrixUtils.elementwiseSame(layer.connectionWeights, layer_deser.connectionWeights ) );		
		assertEquals( true, MatrixUtils.elementwiseSame(layer.input, layer_deser.input ) );
		
	}		

}
