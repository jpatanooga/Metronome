package tv.floe.metronome.math;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;



/**
 *  Collection of Matrix operations
 * 
 * @author josh
 *
 */
public class MatrixUtils {

	/**
	 * Unrolls / flattens a Matrix into a 1 row Matrix
	 * 
	 * @param d
	 * @return
	 */
	public static Matrix unroll(Matrix d) {
		
		Matrix ret = new DenseMatrix( 1, MatrixUtils.length( d ) );
		
		for (int i = 0; i < MatrixUtils.length( d ); i++) {
			
			//ret.put(i,d.get(i));
			//ret.set(i, arg1)
			MatrixUtils.setElement(ret, i, MatrixUtils.getElement(d, i));
			
		}
		
		return ret;
	}

	
	/**
	 * For each column, sum all the values in the column
	 * 
	 * input: N rows x M column matrix
	 * 
	 * output: 1 row, M columns
	 * 
	 */	
	public static Matrix columnSums(Matrix m) {

		Matrix ret_col_sums = new DenseMatrix(1, m.numCols());

		ret_col_sums.assign(0.0);

		// sum into 1 row matrix
		for ( int r = 0; r < m.numRows(); r++ ) {
			for ( int c = 0; c < m.numCols(); c++ ) {
				double val = ret_col_sums.get(0, c); 
				ret_col_sums.set(0, c, m.get(r, c) + val );
			}
		}


		return ret_col_sums;


	}	
	

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

    public static Matrix tanh(Matrix m) {
        for(int i = 0; i < m.numRows(); i++ ) {
            for(int j = 0; j < m.numCols(); j++) {
                m.set(i,j,Math.tanh(m.get(i,j)));
            }
        }
        return m;
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
	 * Computes each row as a sum and outputs a column vector of the sums
	 * 
	 * @param m
	 * @return
	 */
	public static Matrix rowSums(Matrix m) {

		Matrix ret_row_sums = new DenseMatrix(m.numRows(), 1);
		//int col_count = m.numCols();

		//System.out.println("col count: " + col_count);;

		ret_row_sums.assign(0.0);

		// sum into 1 row matrix
		for ( int r = 0; r < m.numRows(); r++ ) {
			for ( int c = 0; c < m.numCols(); c++ ) {

				double row_sum = ret_row_sums.get(r, 0);

				ret_row_sums.set(r, 0, m.get(r, c) + row_sum );
			}
		}


		return ret_row_sums;


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

	/**
	 * Computes the mean value of all elements in the matrix, that is, x.sum() / x.length.
	 * 
	 * @param m
	 * @return
	 */
	public static double mean(Matrix m) {

		double ret = sum(m);
		int count = m.numCols() * m.numRows();

		return ret / (double)count;
	}

	/**
	 * Returns sum of all elements in Matrix
	 * 
	 * @param m
	 * @return
	 */
	public static double sum(Matrix m) {

		double ret = 0;


		for ( int r = 0; r < m.numRows(); r++ ) {
			for ( int c = 0; c < m.numCols(); c++ ) {

				ret += m.get(r, c);

			}
		}		

		return ret;

	}

	/**
	 * Syntactic Sugar for (1 - MatrixOther)
	 * 
	 * @param m
	 * @return
	 */
	public static Matrix oneMinus(Matrix m) {

		return MatrixUtils.ones(m.numRows(), m.numCols()).minus(m);

	}
	
	public static Matrix oneDiv(Matrix m) {
		
		Matrix ones = MatrixUtils.ones(m.numRows(), m.numCols());
		Matrix ret = m.clone();

		for ( int r = 0; r < m.numRows(); r++ ) {

			for ( int c = 0; c < m.numCols(); c++ ) {

//				ret.set(r,  c, Math.log(m.get(r, c)));
				if ( m.get(r, c) == 0.0) {
					ret.set( r, c, 0.01 );
				}

			}

		}
		
		return MatrixUtils.div(ones, ret);
		
		
	}


	/**
	 * Applies the natural logarithm function element-wise on this matrix. 
	 * 
	 * 
	 * @param m
	 * @return
	 */
	public static Matrix log(Matrix m) {

		Matrix ret = m.like();

		for ( int r = 0; r < m.numRows(); r++ ) {

			for ( int c = 0; c < m.numCols(); c++ ) {

				ret.set(r,  c, Math.log(m.get(r, c)));

			}

		}

		return ret;
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


		Matrix m1 = MatrixUtils.neg( m );


		Matrix m2 = MatrixUtils.exp( m1 );


		Matrix denom = ones.plus( m2 );
		Matrix out = MatrixUtils.div(ones, denom);



		return out;
	}
	
	/**
	 * Finds the max value in the input matrix
	 * 
	 * @param m
	 * @return
	 */
	public static double max(Matrix m) {
		
		double max_tmp = m.viewRow(0).maxValue();
		
		for ( int r = 0; r < m.rowSize(); r++ ) {
			
			if (m.viewRow(r).maxValue() > max_tmp ) {
				max_tmp = m.viewRow(r).maxValue();
			}
			
		}
		
		return max_tmp;
		
	}
	
	/**
	 * Finds the max value in the input matrix
	 * 
	 * @param m
	 * @return
	 */
	public static double min(Matrix m) {
		
		double min_tmp = m.viewRow(0).minValue();
		
		for ( int r = 0; r < m.rowSize(); r++ ) {
			
			if (m.viewRow(r).minValue() < min_tmp ) {
				min_tmp = m.viewRow(r).minValue();
			}
			
		}
		
		return min_tmp;
		
	}	
	
	
	/**
	 * Returns a new matrix with softmax function applied to input Matrix
	 * 
	 * @param m
	 * @return
	 */
	public static Matrix softmax(Matrix m) {
		
		 double max = max(m);
         double sum = 0.0;
         
 		Matrix ret = m.like();

 		for ( int r = 0; r < m.numRows(); r++ ) {

 			for ( int c = 0; c < m.numCols(); c++ ) {

 				ret.set(r,  c, Math.exp( m.get(r, c) - max ));

 			}

 		}
         
         sum = sum(ret);

        for ( int r = 0; r < m.numRows(); r++ ) {

 			for ( int c = 0; c < m.numCols(); c++ ) {

 				ret.set(r,  c, ret.get(r, c) / sum );

 			}

 		}
         
         
         return ret;		
		
	}
	

	/**
	 * Negate each elemente in the Matrix
	 * 
	 * @param m - input matrix
	 * @return
	 */
	public static Matrix neg(Matrix m) {

		Matrix ret = m.like();
		
		for ( int r = 0; r < m.numRows(); r++ ) {
			for ( int c = 0; c < m.numCols(); c++ ) {


				double cell = m.get(r, c);


				ret.set(r, c, -1.0 * cell  );
			}
		}

		return ret;

	}

	/**
	 * Applies Math.exp() function to each element of the vector in-place
	 * 
	 * @param m
	 */
	public static Matrix exp(Matrix m) {

		Matrix ret = m.like();		
		
		for (int r = 0; r < m.numRows(); r++) {
			for ( int c = 0; c < m.numCols(); c++ ) {

				ret.set(r, c, (double) Math.exp( m.get(r, c) ) );

			}
		}

		return ret;
	}

	/**
	 * Applies Math.sqrt() function to each element of the vector,
	 * returning the new Matrix with the sqrt'd values
	 * 
	 * @param m
	 */
	public static Matrix sqrt(Matrix m) {

		Matrix ret = m.like();		
		
		for (int r = 0; r < m.numRows(); r++) {
			
			for ( int c = 0; c < m.numCols(); c++ ) {

			//	ret.set(r, c, (double) Math.exp( m.get(r, c) ) );
				ret.set(r, c, (double) Math.sqrt( m.get(r, c) ) );

			}
			
		}

		return ret;
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
	 * @param rows the number of rows
     * @param cols the number of columns
	 * @return
	 */
	public static Matrix ones(int rows, int cols) {

		Matrix ret = new DenseMatrix(rows, cols);
		ret.assign(1.0);

		return ret;
	}
	
	public static Matrix uniform(RandomGenerator rng, int rows, int cols) {
	
		UniformRealDistribution uDist = new UniformRealDistribution(rng,0,1);
		Matrix U = new DenseMatrix(rows, cols);
		for (int r = 0; r < rows; r++) {
			for ( int c = 0; c < cols; c++ ) {
				U.set(r, c, uDist.sample());
			}
		}
		
		return U;
		
	}
	

	/**
	 * Generate a binomial distribution based on the given random number generator,
	 * a matrix of p values, and a max number.
	 * 
	 * 
	 * @param pValues the p matrix to use
	 * @param max the n to use
	 * @param rndNumberGenerator the rng to use
	 * @return a binomial distribution based on the one n, the passed in p values, and rng
	 */	
	public static Matrix genBinomialDistribution(Matrix pValues, int max, RandomGenerator rndNumberGenerator) {

		Matrix dist = pValues.like(); 

		for (int r = 0; r < dist.numRows(); r++) {
			for ( int c = 0; c < dist.numCols(); c++ ) {

				dist.set(r, c, getBinomial(rndNumberGenerator, max, pValues.get(r,  c) ) );

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
	 * @param p
	 * @return
	 */
	public static int getBinomial(RandomGenerator rng, int max, double p) {

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

	/**
	 * Generating numbers in a poisson distribution based on how Knuth says to do it:
	 * 
	 * http://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
	 * 
	 * 
	 * @param lambda
	 * @return
	 */
	public static int getPoisson(double lambda) {

		double L = Math.exp(-lambda);
		double p = 1.0;
		int k = 0;

		do {

			k++;
			p *= Math.random();

		} while (p > L);

		return k - 1;
	}


	public static void assertSameLength(Matrix a, Matrix b) {
		int aLength = a.columnSize() * a.rowSize();
		int bLength = b.columnSize() * b.rowSize();
		if(aLength != bLength)
			throw new IllegalArgumentException("Your first matrix must have the same length as your second");

	}

	public static Matrix elementWiseMultiplication(Matrix a,Matrix b) {
		assertSameLength(a,b);
		Matrix ret = new DenseMatrix(a.rowSize(),a.columnSize());
		Matrix mult = a.rowSize() != b.rowSize() ? b.transpose() : b;
			for(int i = 0; i < ret.rowSize(); i++) {
				for(int j = 0; j < ret.columnSize(); j++) {
					ret.setQuick(i, j, a.getQuick(i, j) * mult.getQuick(i, j));
				}

			}



		return ret;
	}


	/**
	 * Add a row to all rows of the matrix
	 * 
	 * @param m
	 * @param row
	 * @return a new matrix with the target vector row added to each row
	 */
	public static Matrix addRowVector(Matrix m, Vector row) {

		Matrix ret = m.like();
		ret.assign(m);

		for (int r = 0; r < m.numRows(); r++) {

			ret.viewRow(r).assign(row, Functions.PLUS);

		}


		return ret;

	}
	
	public static void ensureValidOutcomeMatrix(Matrix m) {

		boolean found = false;
		
		for (int r = 0; r < m.numRows(); r++) {
			
			for ( int c = 0; c < m.numCols(); c++ ) {

				 if ( m.get(r,  c) > 0 ) {
					 found = true;
					 break;
				 }

			}
		}
		
		if (!found) {
			throw new IllegalStateException("Found a matrix without an outcome");
		}
		
	}
	
	/**
	 * Returns total number of elements in matrix
	 * 
	 * @param m
	 * @return
	 */
	public static int length(Matrix m) {
		
		return m.numRows() * m.numCols();
		
	}
	
	
	/**
	 * Returns a new matrix with Math.pow() applied to each element in the src matrix
	 * 
	 * @param m
	 * @param d
	 * @return
	 */
	public static Matrix pow(Matrix m, double d) {
		
		Matrix ret = m.like();
		
		for (int r = 0; r < m.numRows(); r++) {
			for ( int c = 0; c < m.numCols(); c++ ) {
				ret.set(r, c, (double)Math.pow(m.get(r, c), d) );
			}
		}
		
		return ret;
		
     }
		     
	
	/**
	 * Returns the element as if the matrix was 'unrolled' linearly
	 * 
	 * @param index
	 * @return
	 */
	public static double getElement(Matrix m, int index) {
		
		int rowIndex = index / m.numCols();
		int colIndex = index % m.numCols();
		
		return m.getQuick(rowIndex, colIndex);
		
	}
	

	public static void setElement(Matrix m, int index, double value) {
		
		int rowIndex = index / m.numCols();
		int colIndex = index % m.numCols();
		
		m.setQuick(rowIndex, colIndex, value);
		
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

	public static void debug_print_matrix_stats(Matrix m, String label) {

		System.out.println("Print Matrix Stats [" + label + "] ------- ");
		System.out.println("Rows: " + m.numRows() + ", Cols: " + m.numCols() );
		System.out.println("Print Matrix Stats ------- ");

	}
	
	public static void debug_print_row(Matrix m, int index) {


		//Vector v = m.viewRow(index);
		
		System.out.println("Print Matrix Row ------- ");
		//for (int r = 0; r < m.numRows(); r++) {
			for ( int c = 0; c < m.numCols(); c++ ) {

				//ret.set(r, c, numerator.get(r, c) / denominator.get(r, c)  );
				System.out.print(" " + m.get(index, c));
			}
			System.out.println(" ");
		//}
		System.out.println("Print Matrix Row ------- ");


	}		

	
	public static int iamax(Vector vec) {
		
		//double max = vec.get(0);
		int index = 0;
		
		for ( int x = 0; x < vec.size(); x++ ) {
			
			if ( Math.abs(vec.get(x)) > Math.abs( vec.get(index) ) ) {
				
				index = x;
				
			}
			
			
		}
		
		return index;
		
	}
	
	/**
	 * Returns a copy of this matrix where all elements are set to their absolute values.
	 * 
	 * @param m
	 * @return
	 */
	public static Matrix abs(Matrix m) {
		
		Matrix out = m.like();
		
		for (int r = 0; r < m.numRows(); r++) {
			
			for ( int c = 0; c < m.numCols(); c++ ) {
				
				out.set(r, c, Math.abs( m.get(r, c) ) );

			}
		
		}

		return out;
		
	}
	
	public static Matrix viewRowAsMatrix(Matrix m, int rowIndex) {
		
		return m.viewPart( rowIndex, 1, 0, m.numCols() );		
		
	}
	

	public static Matrix toOutcomeVector(int index,int numOutcomes) {
		int[] nums = new int[numOutcomes];
		nums[ index ] = 1;
		//return toMatrix(nums);
		Matrix out = new DenseMatrix(1, numOutcomes);
		out.set(0, index, 1.0);
		return out;
	}
	
	
	public static Matrix toMatrix(int[] arr) {
		
		Matrix d = new DenseMatrix(1, arr.length);
		
		for (int i = 0; i < arr.length; i++) {
		
			d.set(0, i,arr[i]);
		
		}
		
		//d.reshape(1, d.length);
		return d;
		
	}	
	
	public static Matrix toMatrix(int[][] arr) {
		
		Matrix d = new DenseMatrix( arr.length, arr[0].length );
		
		for (int i = 0; i < arr.length; i++) {
			
			for (int j = 0; j < arr[i].length; j++) {
				
				d.set( i, j, arr[ i ][ j ] );
				
			}
			
		}
		
		return d;
	}
	
	public static double[][] fromMatrix(Matrix m) {
		
		double[][] out = new double[ m.numRows() ][ m.numCols() ];
		
		for (int i = 0; i < m.numRows(); i++) {
			
			for (int j = 0; j < m.numCols(); j++) {
				
				out[ i ][ j ] = m.getQuick(i, j);
				
			}
			
		}
		
		return out;
	}


    public static void subi(Matrix m, Matrix subtractionMatrix) {

        for (int r = 0; r < m.numRows(); r++) {

            for ( int c = 0; c < m.numCols(); c++ ) {

                m.set(r, c, m.get(r, c) - subtractionMatrix.get(r, c) );

            }

        }


    }

	public static void addi(Matrix m, Matrix additionMatrix) {
		
		for (int r = 0; r < m.numRows(); r++) {

			for ( int c = 0; c < m.numCols(); c++ ) {
				
				m.set(r, c, m.get(r, c) + additionMatrix.get(r, c) );
			
			}
			
		}
		
		
	}
	

	public static void subi(Matrix m, double val) {
		
		for (int r = 0; r < m.numRows(); r++) {

			for ( int c = 0; c < m.numCols(); c++ ) {
				
				m.set(r, c, m.get(r, c) - val );
			
			}
			
		}
		
		
	}

	public static void divi(Matrix m, double val) {
		
		for (int r = 0; r < m.numRows(); r++) {

			for ( int c = 0; c < m.numCols(); c++ ) {
				
				m.set(r, c, m.get(r, c) / val );
			
			}
			
		}
		
		
	}
	
	
	public static Matrix normalize( Matrix input) {
		
		//double min = input.min();
		double min = MatrixUtils.min( input );
		
		//double max = input.max();
		double max = MatrixUtils.max( input );
		
		//return input.subi(min).divi(max - min);
		subi( input, min );
		divi( input, max - min );
				
		return input;
	}

	/**
	 * Compares all elements in two Matrices to see if all elements are the same
	 * 
	 * 
	 */	
	public static boolean elementwiseSame(Matrix a, Matrix b) {

		if (a.numRows() != b.numRows() ) {
			return false;
		}
		
		if (a.numCols() != b.numCols()) {
			return false;
		}
		
		for ( int r = 0; r < a.numRows(); r++ ) {

			for ( int c = 0; c < a.numCols(); c++ ) {

				if (a.get(r, c) != b.get(r, c)) {
					return false;
				}
				
			}
			
		}


		return true;


	}		

	public static boolean isNaN(Matrix test) {
		
		for (int i = 0; i < MatrixUtils.length( test ); i++) {
		
			if (Double.isNaN( MatrixUtils.getElement( test, i ) ) ) {
				
				return true;
				
			}
			
		}
		
		return false;
		
	}	
	





}
