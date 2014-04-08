package tv.floe.metronome.math;

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

}
