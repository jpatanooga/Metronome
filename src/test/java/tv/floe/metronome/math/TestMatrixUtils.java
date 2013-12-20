package tv.floe.metronome.math;


import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

import org.junit.Test;



//import tv.floe.ml_workbench.jblas.MatrixDebug;

public class TestMatrixUtils {

	
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

		
	}
}
