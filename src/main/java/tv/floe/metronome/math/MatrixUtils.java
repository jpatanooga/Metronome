package tv.floe.metronome.math;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 *  
 * 
 * @author josh
 *
 */
public class MatrixUtils {
	
	
	/**
	 * For each column, sum all the values in the column and average them together
	 * 
	 * input: N rows x M column matrix
	 * 
	 * output: 1 row, M columns
	 * 
	 */	
	public static Matrix columnMeans(Matrix m) {
		
		//m.aggregateRows(arg0)
		Matrix ret_col_means = new DenseMatrix(1, m.numCols());
		int row_count = m.numRows();
		
		ret_col_means.assign(0.0);

		// sum into 1 row matrix
		for ( int r = 0; r < m.numRows(); r++ ) {
			for ( int c = 0; c < m.numCols(); c++ ) {
				double val = ret_col_means.get(0, c); 
				ret_col_means.set(0, c, m.get(r, c) + val );
			}
		}

		// average
		for ( int c = 0; c < m.numCols(); c++ ) {
			double val = ret_col_means.get(0, c); 
			ret_col_means.set(0, c, val / row_count );
		}
		
		return ret_col_means;
		
		
	}
	
	public static Matrix rowMeans(Matrix m) {
		
		Matrix ret_row_means = new DenseMatrix(m.numRows(), 1);
		int col_count = m.numCols();
		
		//System.out.println("col count: " + col_count);;
		
		ret_row_means.assign(0.0);

		// sum into 1 row matrix
		for ( int r = 0; r < m.numRows(); r++ ) {
			for ( int c = 0; c < m.numCols(); c++ ) {
				
				
				double row_sum = ret_row_means.get(r, 0);
				
				//System.out.println("row_sum: " + row_sum + " + " + m.get(r, c));
				
				ret_row_means.set(r, 0, m.get(r, c) + row_sum );
			}
		}

		// average
		for ( int r = 0; r < m.numRows(); r++ ) {
			double row_sum_for_col = ret_row_means.get(r, 0);
			ret_row_means.set(r, 0, row_sum_for_col / col_count );
			
			//System.out.println("row mean: " + (row_sum_for_col / col_count));
		}
		
		return ret_row_means;
				
		
	}

	/**
	 * Computes either a row-wise mean or a column-wise mean on the input Matrix m
	 * 
	 * @param m - Matrix to perform Mean operation on
	 * @param axis - 0 for a column-wise mean, 1 for a row-wise mean
	 * @return
	 */
	public static Matrix mean(Matrix m, int axis) {
		
		// create a vector of r-x-1 size
		Matrix ret = new DenseMatrix(m.rowSize(),1);
		//column wise
		if(axis == 0) {
			//return m.columnMeans();
			return MatrixUtils.columnMeans(m);
		}
		//row wise
		else if(axis == 1) {
			//return ret.rowMeans();
			return MatrixUtils.rowMeans(m);
		}


		return ret;
	}

	public Matrix log(Matrix m) {
		return null;
	}
	
	public Matrix sigmoid(Matrix m) {
		return null;
	}
	
	public Matrix ones(Matrix m) {
		return null;
	}
	
	public Matrix binomial(Matrix m) {
		return null;
	}
	
	public void addRowVector(Matrix m, Vector row) {
		
	}
	
	
	
	
	
	
}
