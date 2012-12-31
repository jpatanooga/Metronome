package tv.floe.metronome.linearregression;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import tv.floe.metronome.utils.Utils;

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
	
	@Test
	public void TestAvg() {
		
		Vector v = new DenseVector(2);
		v.set(0, 0.1);
		v.set(1, 0.1);

		Vector v2 = new DenseVector(2);
		v2.set(0, 0.40);
		v2.set(1, 0.20);
		
		Vector v_out = v.plus(v2);
		
		Utils.PrintVector(v_out);
		
		Vector vec_d = v_out.divide(2);
	
		Utils.PrintVector(vec_d);
		
		Vector v_assign = new DenseVector(2);
		v_assign.set(0, 20);
		v_assign.assign(vec_d);
		
		Utils.PrintVector(v_assign);
		
	}

}
