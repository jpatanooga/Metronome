package tv.floe.metronome.math;

import org.apache.commons.math3.random.RandomGenerator;
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
	
	/**
	 * Calculate the sigmoid function in place over a matrix
	 * 
	 * @param m
	 * @return
	 */
	public static Matrix sigmoid(Matrix m) {
		
//		DoubleMatrix ones = DoubleMatrix.ones(x.rows, x.columns);
//	    return ones.div(ones.add(MatrixFunctions.exp(x.neg())));
		
		Matrix ones = new DenseMatrix(m.numRows(), m.numCols());
		ones.assign(1.0);
		
		
		MatrixUtils.neg( m );
		
		
		MatrixUtils.exp( m );
		
		
		Matrix denom = ones.plus( m );
		Matrix out = MatrixUtils.div(ones, denom);
				
		
		
		return out;
	}
	
	/**
	 * Negate each elemente in the Matrix
	 * 
	 * @param m - input matrix
	 * @return
	 */
	public static void neg(Matrix m) {
		
		for ( int r = 0; r < m.numRows(); r++ ) {
			for ( int c = 0; c < m.numCols(); c++ ) {
				
				
				double cell = m.get(r, c);
				
				
				m.set(r, c, -1.0 * cell  );
			}
		}
		
		//return m;

	}
	
	/**
	 * Applies Math.exp() function to each element of the vector in-place
	 * 
	 * @param m
	 */
	public static void exp(Matrix m) {
		
		for (int r = 0; r < m.numRows(); r++) {
			for ( int c = 0; c < m.numCols(); c++ ) {
		
				m.set(r, c, (double) Math.exp( m.get(r, c) ) );
				
			}
		}
        
		//return m;
	}
		
	
	
	/**
	 * Elementwise divide by a matrix (in place).
	 * 
	 * @param m
	 * @return
	 */
	public static Matrix div(Matrix numerator, Matrix denominator) {
		
		Matrix ret = numerator.like();
		
		for (int r = 0; r < numerator.numRows(); r++) {
			for ( int c = 0; c < numerator.numCols(); c++ ) {
		
				ret.set(r, c, numerator.get(r, c) / denominator.get(r, c)  );
				
			}
		}
		
		return ret;
	}
	
	/**
	 * Creates a new matrix in which all values are equal 1.
	 * 
	 * @param m
	 * @return
	 */
	public static Matrix ones(int rows, int cols) {
		
		Matrix ret = new DenseMatrix(rows, cols);
		ret.assign(1.0);
		
		return ret;
	}
	
	/**
	 * Generate a binomial distribution based on the given random number generator,
	 * a matrix of p values, and a max number.
	 * 
	 * 
	 * @param p the p matrix to use
	 * @param n the n to use
	 * @param rng the rng to use
	 * @return a binomial distribution based on the one n, the passed in p values, and rng
	 */	
	public static Matrix genBinomialDistribution(Matrix pValues, int max, RandomGenerator rndNumberGenerator) {

		Matrix dist = pValues.like(); 
		
		//for(int i = 0; i < ret.length; i++) {
			//ret.put(i,MathUtils.binomial(rng, n, p.get(i)));
		//}
		
		for (int r = 0; r < dist.numRows(); r++) {
			for ( int c = 0; c < dist.numCols(); c++ ) {
		
				dist.set(r, c, binomial(rndNumberGenerator, max, pValues.get(r,  c) ) );
				
			}
		}
		
		
		return dist;
		
	
	}
	
	/**
	 * Generates a binomial distributed number using the given random number generator
	 * 
	 * Based on: http://luc.devroye.org/chapter_ten.pdf
	 * 
	 * @param rng
	 * @param n
	 * @param p
	 * @return
	 */
	public static int binomial(RandomGenerator rng, int max, double p) {
		
		if (p < 0 || p > 1) {
			return 0;
		}
		
		int c = 0;
		double r;
		
		for (int i = 0; i < max; i++) {
			
			r = rng.nextDouble();
			
			if (r < p) { 
				c++;
			}
			
		}
		
		return c;
	}	
	
	public void addRowVector(Matrix m, Vector row) {
		
	}
	
	public static void debug_print(Matrix m) {
		
		
		System.out.println("Print Matrix ------- ");
		for (int r = 0; r < m.numRows(); r++) {
			for ( int c = 0; c < m.numCols(); c++ ) {
		
				//ret.set(r, c, numerator.get(r, c) / denominator.get(r, c)  );
				System.out.print(" " + m.get(r, c));
			}
			System.out.println(" ");
		}
		System.out.println("Print Matrix ------- ");
		
		
	}	
	
	
	
	
	
	
}
