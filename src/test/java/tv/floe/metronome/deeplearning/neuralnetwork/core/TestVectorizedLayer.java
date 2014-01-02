package tv.floe.metronome.deeplearning.neuralnetwork.core;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

import tv.floe.metronome.math.MatrixUtils;

public class TestVectorizedLayer {

	public static Matrix buildTestInputDataset() {
		
		double[][] data = new double[][]
				{
					{1,1,1},
					{1,1,1},
					{0,0,1}
				};
		
		Matrix d = new DenseMatrix(data);

		return d;
		
	}	
	
	@Test
	public void testBuildLayer( ) {
		
		RandomGenerator rnd = new MersenneTwister(1234);
		
		LayerVectorized layer = new LayerVectorized(3, 2, rnd);
		
		assertEquals(3, layer.connectionWeights.numRows() );
		assertEquals(2, layer.connectionWeights.numCols() );
		
		
	}
	
	
	@Test
	public void testComputeActivations() {
		
		Matrix input = buildTestInputDataset();
		
		RandomGenerator rnd = new MersenneTwister(1234);
		
		LayerVectorized layer = new LayerVectorized(3, 2, rnd);

		layer.setInput(input);
		
		Matrix m = layer.computeActivationOutput();
		
		MatrixUtils.debug_print_matrix_stats(m, "activations");
		
		MatrixUtils.debug_print(m);
		
		
		
		
		
		
		
		
		
	}

}
