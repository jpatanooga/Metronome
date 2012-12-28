package tv.floe.metronome.linearregression;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.junit.Test;

/**
 * In the case of linear regression this is simply the dot product of the 
 * current instance and the current parameter vector
 * 
 * 
 * 
 * @author josh
 *
 */
public class TestHypothesisFunction {

	@Test
	public void testMahoutLinearCombinaton() {
		
		System.out.println( "> testMahoutLinearCombinaton" );
		
		Matrix m = new DenseMatrix( 1, 2 );
		m.set(0, 0, 0.2);
		m.set(0, 1, 0.3);
		//m.set(1, 0, 0.4);
		//m.set(1, 1, 0.5);
		
		Vector v = new DenseVector(2);
		v.set(0, 0.1);
		v.set(1, 0.1);
		//v.set(2, 0.1);
		
		// returns the dot product for each row in the matrix as a vector
		Vector v2 = m.times(v);
		
		//System.out.println(">>>>>> " + v.size());
		System.out.println("> " + m.get(0, 0) );
		System.out.println("> " + m.get(0, 1) );
//		System.out.println("> " + m.get(1, 0) );
//		System.out.println("> " + m.get(1, 1) );
		
		System.out.println("> Vector out ------ " );
		System.out.println("> size: " + v2.size() );
		System.out.println("> c0: " + v2.get(0) );
//		System.out.println("> c1: " + v2.get(1) );
		
	}
	
}
