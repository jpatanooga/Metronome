package tv.floe.metronome.deeplearning.rbm;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

import tv.floe.metronome.math.MatrixUtils;

/**
 * Base test for both
 * - RBMOptimizer
 * - NNOptimizer
 * 
 * Given that NNOptimizer is abstract but implemenets some methods we need to test out
 * 
 * 
 * @author josh
 *
 */
public class TestRestrictedBoltzmannMachineOptimizer {

	public static Matrix buildTestInputDataset() {
		
		double[][] data = new double[][]
				{
					{1,1,1,0,0,0},
					{1,0,1,0,0,0},
					{1,1,1,0,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,1,0}
				};
		
		Matrix d = new DenseMatrix(data);

		return d;
		
	}	
	
	@Test
	public void testParameterBuffers() {
		
		
		Matrix input = buildTestInputDataset();
		
		double[] buffer_stock = new double[20];
		
		double[] buffer_zeros = new double[20];
		for ( int x = 0; x < buffer_zeros.length; x++ ) {
			buffer_zeros[x] = x;
		}

		
		RestrictedBoltzmannMachine rbm_stock = new RestrictedBoltzmannMachine(6, 2, null);
		rbm_stock.setInput(input);

		rbm_stock.trainTillConvergence(0.01, 1, input);
		
		System.out.println( "Parameters: " + rbm_stock.optimizer.getNumParameters() );
		
		assertEquals(20, rbm_stock.optimizer.getNumParameters());
		
		rbm_stock.optimizer.getParameters(buffer_stock);
		
		for ( int x = 0; x < buffer_stock.length; x++ ) {
			System.out.println( "buffer[" + x + "]: " + buffer_stock[x] );
		}
		
		rbm_stock.optimizer.setParameters(buffer_zeros);
		
		rbm_stock.optimizer.getParameters(buffer_stock);

		for ( int x = 0; x < buffer_stock.length; x++ ) {
			System.out.println( "buffer[" + x + "]: " + buffer_stock[x] );
		}
		
		assertEquals( 0, rbm_stock.optimizer.getParameter(0), 0.0 );
		assertEquals( 5, rbm_stock.optimizer.getParameter(5), 0.0 );
		assertEquals( 10, rbm_stock.optimizer.getParameter(10), 0.0 );
		assertEquals( 14, rbm_stock.optimizer.getParameter(14), 0.0 );

		
	}
	
	
	/*
	@Test
	public void testOptimizerVsStockCDkImplMatch() {

		Matrix input = buildTestInputDataset();
		
		double[] buffer_stock = new double[20];
		RestrictedBoltzmannMachine rbm_stock = new RestrictedBoltzmannMachine(6, 2, null);
		rbm_stock.setInput(input);
		
		//MatrixUtils.debug_print( rbm_stock.connectionWeights );

		rbm_stock.setupCDkDebugBuffer(buffer_stock);
		rbm_stock.contrastiveDivergence(0.1, 1, input);
		
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null);
		rbm.setInput(input);
		
		
		double[] buffer = new double[20];
		
		rbm.optimizer = new RestrictedBoltzmannMachineOptimizer(rbm, 0.1, new Object[]{1});
		rbm.optimizer.getValueGradient(buffer);
		
		//Matrix b = new DenseMatrix(1, buffer.length);
		//b.viewRow(0).assign(buffer);
		
		//MatrixUtils.debug_print(b);
		
		// TODO: check that the optimizer works the same as stock CDk
		
		for ( int x = 0; x < buffer.length; x++ ) {
			
			assertEquals(buffer[x], buffer_stock[x], 0.0);
			
		}
		
		
	}
	
	*/

}
