package tv.floe.metronome.math;


import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import org.junit.Test;

public class TestMatrixUtils {

	@Test
	public void testSumAndMean() {
		
		Matrix m = new DenseMatrix(2, 3);
		m.setQuick(0, 0, 6);
		m.setQuick(0, 1, 9);
		m.setQuick(0, 2, 3);

		m.setQuick(1, 0, 5);
		m.setQuick(1, 1, 10);
		m.setQuick(1, 2, 0);

		double sum = MatrixUtils.sum(m);
		
		double mean = MatrixUtils.mean(m);
		
		assertEquals(33.0, sum, 0.0);
		
		assertEquals( 33.0 / 6.0, mean, 0.0 );
		
		
	}
	
	/**
	 * 
	 * input: N rows x M column matrix
	 * 
	 * output: N row, 1 columns
	 * 
	 * a single column is output, and the row entry for each one contains the average of the row
	 * 
	 * 
	 */
	@Test
	public void testRowMeans() {

		System.out.println("------- testRowMeans ------- ");
		
		Matrix m = new DenseMatrix(2, 3);
		m.setQuick(0, 0, 6);
		m.setQuick(0, 1, 9);
		m.setQuick(0, 2, 3);

		m.setQuick(1, 0, 5);
		m.setQuick(1, 1, 10);
		m.setQuick(1, 2, 0);
		
		
		Matrix row_means_matrix = MatrixUtils.rowMeans(m); //m.rowMeans();
		
		assertEquals(1, row_means_matrix.numCols() );
		assertEquals(m.numRows(), row_means_matrix.numRows() );
		
		assertEquals( 6.0, row_means_matrix.get(0, 0), 0.0);
		assertEquals( 5.0, row_means_matrix.get(1, 0), 0.0);
		
		Matrix row_means_matrix_2 = MatrixUtils.mean(m, 1);

		assertEquals(1, row_means_matrix_2.numCols() );
		assertEquals(m.numRows(), row_means_matrix_2.numRows() );
		
		assertEquals( 6.0, row_means_matrix_2.get(0, 0), 0.0);
		assertEquals( 5.0, row_means_matrix_2.get(1, 0), 0.0);

	}		

	@Test
	public void testColumnMeans() {

		System.out.println("------- testColMeans ------- ");
		
		Matrix m = new DenseMatrix(2, 3);
		m.setQuick(0, 0, 2);
		m.setQuick(0, 1, 3);
		m.setQuick(0, 2, 5);

		m.setQuick(1, 0, 3);
		m.setQuick(1, 1, 4);
		m.setQuick(1, 2, 6);
		
		
		Matrix col_means_matrix = MatrixUtils.columnMeans(m); //m.rowMeans();
		

		assertEquals(1, col_means_matrix.numRows() );
		assertEquals(m.numCols(), col_means_matrix.numCols() );

		assertEquals( 2.5, col_means_matrix.get(0, 0), 0.0);
		assertEquals( 3.5, col_means_matrix.get(0, 1), 0.0);
		assertEquals( 5.5, col_means_matrix.get(0, 2), 0.0);

		
		Matrix col_mean_matrix_2 = MatrixUtils.mean(m, 0);

		assertEquals(1, col_mean_matrix_2.numRows() );
		assertEquals(m.numCols(), col_mean_matrix_2.numCols() );

		assertEquals( 2.5, col_mean_matrix_2.get(0, 0), 0.0);
		assertEquals( 3.5, col_mean_matrix_2.get(0, 1), 0.0);
		assertEquals( 5.5, col_mean_matrix_2.get(0, 2), 0.0);
		
		
		
	}
	
	@Test
	public void testMatrixRowSums() {
		
		System.out.println("------- testRowSums ------- ");
		
		Matrix m = new DenseMatrix(2, 3);
		
		m.setQuick(0, 0, 6);
		m.setQuick(0, 1, 9);
		m.setQuick(0, 2, 3);

		m.setQuick(1, 0, 5);
		m.setQuick(1, 1, 10);
		m.setQuick(1, 2, 0);
		
		
		Matrix row_sums_matrix = MatrixUtils.rowSums(m);
		
		assertEquals(1, row_sums_matrix.numCols() );
		assertEquals(m.numRows(), row_sums_matrix.numRows() );
		
		assertEquals( 18.0, row_sums_matrix.get(0, 0), 0.0);
		assertEquals( 15.0, row_sums_matrix.get(1, 0), 0.0);
		
		
	}
	
	@Test
	public void testMatrixOnes() {
		
		Matrix ones_alt = MatrixUtils.ones(3, 4);
		
		Matrix ones = MatrixUtils.ones(3, 4);
		
		assertEquals( 3, ones.numRows());
		assertEquals( 4, ones.numCols());
		
		for ( int r = 0; r < ones.numRows(); r++ ) {
			
			for ( int c = 0; c < ones.numCols(); c++ ) {
				
				assertEquals(1.0, ones.getQuick(r, c), 0.0);
				
			}
			
		}
		
		Matrix twos = ones.plus(ones_alt);

		for ( int r = 0; r < twos.numRows(); r++ ) {
			
			for ( int c = 0; c < twos.numCols(); c++ ) {
				
				assertEquals(2.0, twos.getQuick(r, c), 0.0);
				
			}
			
		}

		
	}

	@Test
	public void testMatrixOneMinus() {

		
		
		Matrix m0 = new DenseMatrix(2, 3);
		m0.setQuick(0, 0, 2);
		m0.setQuick(0, 1, 3);
		m0.setQuick(0, 2, 5);

		m0.setQuick(1, 0, 3);
		m0.setQuick(1, 1, 4);
		m0.setQuick(1, 2, 6);
		
		Matrix oneMinus = MatrixUtils.oneMinus(m0);
		
		assertEquals( -1.0, oneMinus.get(0, 0), 0.0 );
		assertEquals( -2.0, oneMinus.get(0, 1), 0.0 );
		assertEquals( -4.0, oneMinus.get(0, 2), 0.0 );
		
		assertEquals( -2.0, oneMinus.get(1, 0), 0.0 );
		assertEquals( -3.0, oneMinus.get(1, 1), 0.0 );
		assertEquals( -5.0, oneMinus.get(1, 2), 0.0 );

	}
	
	@Test
	public void testMatrixOneDiv() {
		
		Matrix m0 = new DenseMatrix(2, 3);
		m0.setQuick(0, 0, 2);
		m0.setQuick(0, 1, 3);
		m0.setQuick(0, 2, 5);

		m0.setQuick(1, 0, 3);
		m0.setQuick(1, 1, 4);
		m0.setQuick(1, 2, 6);
		
		Matrix oneMinus = MatrixUtils.oneDiv(m0);
		
		assertEquals( 0.5, oneMinus.get(0, 0), 0.0 );
		assertEquals( 0.33333, oneMinus.get(0, 1), 0.001 );
		assertEquals( 0.2, oneMinus.get(0, 2), 0.0 );
		
		assertEquals( 0.33333, oneMinus.get(1, 0), 0.001 );
		assertEquals( 0.25, oneMinus.get(1, 1), 0.0 );
		assertEquals( 1.0/6.0, oneMinus.get(1, 2), 0.0 );
		
		
	}
	
	@Test
	public void testMatrixDiv() {

		
		
		Matrix m0 = new DenseMatrix(2, 3);
		m0.setQuick(0, 0, 2);
		m0.setQuick(0, 1, 3);
		m0.setQuick(0, 2, 5);

		m0.setQuick(1, 0, 3);
		m0.setQuick(1, 1, 4);
		m0.setQuick(1, 2, 6);
		
		Matrix m1 = new DenseMatrix(2, 3);
		m1.setQuick(0, 0, 2);
		m1.setQuick(0, 1, 3);
		m1.setQuick(0, 2, 5);

		m1.setQuick(1, 0, 3);
		m1.setQuick(1, 1, 4);
		m1.setQuick(1, 2, 6);
		
		
		Matrix mDiv = MatrixUtils.div(m0, m1);
		
		for ( int r = 0; r < m0.numRows(); r++ ) {
			
			for ( int c = 0; c < m0.numCols(); c++ ) {
				
				assertEquals(1.0, mDiv.getQuick(r, c), 0.0);
				
			}
			
		}
		
		
	}
	

	@Test
	public void testMatrixNeg() {

		
		
		Matrix m0 = new DenseMatrix(2, 3);
		m0.setQuick(0, 0, 2);
		m0.setQuick(0, 1, 3);
		m0.setQuick(0, 2, 5);

		m0.setQuick(1, 0, 3);
		m0.setQuick(1, 1, 4);
		m0.setQuick(1, 2, 6);
		
		
		
		Matrix m1 = MatrixUtils.neg(m0);

		assertEquals(-2.0, m1.get(0, 0), 0.0);
		assertEquals(-3.0, m1.get(0, 1), 0.0);
		assertEquals(-5.0, m1.get(0, 2), 0.0);

		assertEquals(-3.0, m1.get(1, 0), 0.0);
		assertEquals(-4.0, m1.get(1, 1), 0.0);
		assertEquals(-6.0, m1.get(1, 2), 0.0);
		
		
		
		assertEquals(2.0, m0.get(0, 0), 0.0);
		assertEquals(3.0, m0.get(0, 1), 0.0);
		assertEquals(5.0, m0.get(0, 2), 0.0);

		assertEquals(3.0, m0.get(1, 0), 0.0);
		assertEquals(4.0, m0.get(1, 1), 0.0);
		assertEquals(6.0, m0.get(1, 2), 0.0);
		
		
		
	}	
	
	@Test
	public void testMatrixExp() {

		
		
		Matrix m0 = new DenseMatrix(2, 3);
		m0.setQuick(0, 0, 2);
		m0.setQuick(0, 1, 3);
		m0.setQuick(0, 2, 5);

		m0.setQuick(1, 0, 3);
		m0.setQuick(1, 1, 4);
		m0.setQuick(1, 2, 6);
		
		
		
		Matrix mRet = MatrixUtils.exp(m0);

		//System.out.println("exp: " +  m0.get(0, 0));
		
		assertEquals(Math.exp(2.0), mRet.get(0, 0), 0.0);
		assertEquals(Math.exp(3.0), mRet.get(0, 1), 0.0);
		assertEquals(Math.exp(5.0), mRet.get(0, 2), 0.0);

		assertEquals(Math.exp(3.0), mRet.get(1, 0), 0.0);
		assertEquals(Math.exp(4.0), mRet.get(1, 1), 0.0);
		assertEquals(Math.exp(6.0), mRet.get(1, 2), 0.0);

		
		assertEquals( 2.0, m0.get(0, 0), 0.0);
		assertEquals( 3.0, m0.get(0, 1), 0.0);
		assertEquals( 5.0, m0.get(0, 2), 0.0);

		assertEquals( 3.0, m0.get(1, 0), 0.0);
		assertEquals( 4.0, m0.get(1, 1), 0.0);
		assertEquals( 6.0, m0.get(1, 2), 0.0);
		
		
	}
			
	@Test
	public void testMatrixSigmoid() {

		
		
		Matrix m0 = new DenseMatrix(1, 3);
		m0.setQuick(0, 0, 2);
		m0.setQuick(0, 1, 3);
		m0.setQuick(0, 2, 5);
		
		Matrix sig = MatrixUtils.sigmoid(m0);
		
		assertEquals( (1.0 / (1.0 + (double)Math.exp(-1.0*2.0))), sig.get(0,0), 0.0 );
		assertEquals( (1.0 / (1.0 + (double)Math.exp(-1.0*3.0))), sig.get(0,1), 0.0 );
		assertEquals( (1.0 / (1.0 + (double)Math.exp(-1.0*5.0))), sig.get(0,2), 0.0 );

		
	}

	
	@Test
	public void testMatrixSoftmax() {

		
		
		Matrix m0 = new DenseMatrix(1, 3);
		m0.setQuick(0, 0, 2);
		m0.setQuick(0, 1, 3);
		m0.setQuick(0, 2, 5);
		
		Matrix sig = MatrixUtils.softmax(m0);
		
		double max = 5;
		double val_0_exp_minus_max = Math.exp(2 - max);
		double val_1_exp_minus_max = Math.exp(3 - max);
		double val_2_exp_minus_max = Math.exp(5 - max);
		
		double sum = val_0_exp_minus_max + val_1_exp_minus_max + val_2_exp_minus_max;
		
		
		assertEquals( val_0_exp_minus_max / sum, sig.get(0,0), 0.0 );
		assertEquals( val_1_exp_minus_max / sum, sig.get(0,1), 0.0 );
		assertEquals( val_2_exp_minus_max / sum, sig.get(0,2), 0.0 );

		
	}	
	
	@Test
	public void testMatrixLog() {

		
		
		Matrix m0 = new DenseMatrix(1, 3);
		m0.setQuick(0, 0, 2);
		m0.setQuick(0, 1, 3);
		m0.setQuick(0, 2, 5);
		
		Matrix logs = MatrixUtils.log(m0);
		
		assertEquals( Math.log(2), logs.get(0,0), 0.0 );
		assertEquals( Math.log(3), logs.get(0,1), 0.0 );
		assertEquals( Math.log(5), logs.get(0,2), 0.0 );

		
	}
	
	
	@Test
	public void testMatrixBinomialGeneration() {

		RandomGenerator g = new MersenneTwister(123);
		
		Matrix m0 = new DenseMatrix(1, 3);
		m0.setQuick(0, 0, 0.5);
		m0.setQuick(0, 1, 0.5);
		m0.setQuick(0, 2, 0.1);
		
		for ( int x = 0; x < 10; x++ ) {
			Matrix bin = MatrixUtils.genBinomialDistribution(m0, 1, g);
			
			MatrixUtils.debug_print(bin);

		}
		
	}
	
	
	@Test
	public void testMatrixAddRowVector() {

		
		Matrix m0 = new DenseMatrix(2, 3);
		m0.setQuick(0, 0, 2);
		m0.setQuick(0, 1, 3);
		m0.setQuick(0, 2, 5);

		m0.setQuick(1, 0, 3);
		m0.setQuick(1, 1, 4);
		m0.setQuick(1, 2, 6);
	
		Vector v0 = new DenseVector(3);
		v0.setQuick(0, 1);
		v0.setQuick(1, 2);
		v0.setQuick(2, 3);
		
		Matrix m1 = MatrixUtils.addRowVector(m0, v0);
		
		
		assertEquals(3.0, m1.get(0, 0), 0.0);
		assertEquals(5.0, m1.get(0, 1), 0.0);
		assertEquals(8.0, m1.get(0, 2), 0.0);

		assertEquals(4.0, m1.get(1, 0), 0.0);
		assertEquals(6.0, m1.get(1, 1), 0.0);
		assertEquals(9.0, m1.get(1, 2), 0.0);

		
	}	
	
	@Test
	public void testUniform() {
		
		RandomGenerator g = new MersenneTwister(123);
		
		Matrix u = MatrixUtils.uniform(g, 3, 4);
		
		assertEquals(3, u.numRows());
		assertEquals(4, u.numCols());
		
	}
	
	
	
}
