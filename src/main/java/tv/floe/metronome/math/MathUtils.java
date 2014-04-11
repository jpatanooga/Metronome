package tv.floe.metronome.math;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

public class MathUtils {

	/**
	 * Normalize a value
	 * (val - min) / (max - min)
	 * @param val value to normalize
	 * @param max max value
	 * @param min min value
	 * @return the normalized value
	 */
	public static double normalize(double val,double min,double max) {
		if(max < min)
			throw new IllegalArgumentException("Max must be greather than min");
		
		return (val - min) / (max - min);
	}
	
	public static double norm1(Matrix m) {
		
        double norm = 0.0;
        
        for (int i = 0; i < MatrixUtils.length( m ); i++) {
            
        	norm += Math.abs( MatrixUtils.getElement(m, i) );
        	
        }
        
        return norm;
    }	
	
	/**
	  * The Euclidean norm of the matrix as vector, also the Frobenius
      * norm of the matrix.
      */
	public static double norm2(Matrix m) {
		
	    double norm = 0.0;
	    
	    for (int i = 0; i < MatrixUtils.length( m ); i++) {
	    	
	        norm += MatrixUtils.getElement(m, i) * MatrixUtils.getElement(m, i);
	        
	    }
	    
	    return (double) Math.sqrt(norm);
	    
	}	
	
	/** Compute scalar product between dx and dy. 
	 *
	 * Source: JBLAS.rdot() impl
	 */
	public static double rdot(int n, double[] dx, int dxIdx, int incx, double[] dy, int dyIdx, int incy) {
        double s = 0.0;
        if (incx == 1 && incy == 1 && dxIdx == 0 && dyIdx == 0) {
            for (int i = 0; i < n; i++)
                s += dx[i] * dy[i];
        }
        else {
            for (int c = 0, xi = dxIdx, yi = dyIdx; c < n; c++, xi += incx, yi += incy) {
                s += dx[xi] * dy[yi];
            }
        }
        return s;
    }	
	
    /**
     * Compute x^T * y (dot product)
     */
    public static double dot(Matrix x, Matrix y) {
        //return rdot(x.length, x.data, 0, 1, y.data, 0, 1);
    	return rdot( MatrixUtils.length( x ), ArrayUtils.flatten( MatrixUtils.fromMatrix( x ) ), 0, 1, ArrayUtils.flatten( MatrixUtils.fromMatrix( y ) ), 0, 1);
    }	
    
    public static Matrix rand(int rows, int columns) {
        Matrix m = new DenseMatrix(rows, columns);

        java.util.Random rndGen = new java.util.Random();
        for ( int r = 0; r < m.numRows(); r++ ) {
        	
        	for ( int c = 0; c < m.numCols(); c++ ) {
        		
        		m.setQuick( r, c, rndGen.nextDouble() );
        		
        	}
        	
        }

        return m;
    }    

}
