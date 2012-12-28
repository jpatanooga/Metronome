package tv.floe.metronome.linearregression;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class TestVectorMathStuff {

	@Test
	public void test() {
		
		Vector a = new DenseVector(3);
		a.set(0, 0.1);
		a.set(1, 0.1);
		a.set(2, 0.1);
		
		Vector b = new DenseVector(3);
		b.set(0, 0.1);
		b.set(1, 0.2);
		b.set(2, 0.3);
		
		Vector c = a.times(b);
		
		double d = a.dot(b);
		
		System.out.println(">>>>>> " + c.size());
		System.out.println("> " + c.get(0) );
		System.out.println("> " + c.get(1) );
		System.out.println("> " + c.get(2) );
		
		System.out.println("> Dot Product: " + d );
		
		//fail("Not yet implemented");
	}
	
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
