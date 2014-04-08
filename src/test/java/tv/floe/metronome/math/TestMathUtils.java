package tv.floe.metronome.math;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

public class TestMathUtils {

	@Test
	public void testNorm1() {
		
		Matrix m = new DenseMatrix(1, 2);
		m.setQuick(0, 0, 6);
		m.setQuick(0, 1, 9);

		Matrix m2 = new DenseMatrix(1, 2);
		m2.setQuick(0, 0, -6);
		m2.setQuick(0, 1, 9);

		double m_ret = MathUtils.norm1(m);
		double m2_ret = MathUtils.norm1(m2);
		
		assertEquals( 15.0, m_ret, 0.0 );
		assertEquals( 15.0, m2_ret, 0.0 );
		
		
	}

	
	@Test
	public void testNorm2() {
		
		Matrix m = new DenseMatrix(1, 2);
		m.setQuick(0, 0, 2);
		m.setQuick(0, 1, 4);

		Matrix m2 = new DenseMatrix(1, 2);
		m2.setQuick(0, 0, -2);
		m2.setQuick(0, 1, 2);

		double m_ret = MathUtils.norm2(m);
		double m2_ret = MathUtils.norm2(m2);
		
		assertEquals( Math.sqrt(20), m_ret, 0.0 );
		assertEquals( Math.sqrt(8), m2_ret, 0.0 );
		
		
	}	
	
	
}
