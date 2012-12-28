package tv.floe.metronome.linearregression;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestSquaredErrorLossFunction {

	@Test
	public void test() {
		//fail("Not yet implemented");
		
		double hypothesis_value = 1;
		double actual = 1;
		
		double error = SquaredErrorLossFunction.Calc(hypothesis_value, actual);
		
		assertEquals( 0.0, error, 0.01 );
		
		System.out.println( "> err: " + error );		
				
	}

}
