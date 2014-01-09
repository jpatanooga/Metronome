package tv.floe.metronome.deeplearning.neuralnetwork.layer;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

import tv.floe.metronome.math.MatrixUtils;


public class TestHiddenLayer {

	@Test
	public void testSimple() {
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

}
