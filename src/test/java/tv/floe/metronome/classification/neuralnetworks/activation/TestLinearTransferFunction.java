package tv.floe.metronome.classification.neuralnetworks.activation;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestLinearTransferFunction {

	@Test
	public void testBasicLinearTransferFunction() {

		Linear fn = new Linear();
		
		fn.setSlope(0.5f);
		
		assertEquals(0.5f, fn.getSlope(), 0.0001f );
		
		double out_der = fn.getDerivative(0.5f);
		
		assertEquals( 0.5f, out_der, 0.0001f );

		
		double output = fn.getOutput(0.5f);
		
		assertEquals( 0.25f, output, 0.0001f );
		
		
	}

}
