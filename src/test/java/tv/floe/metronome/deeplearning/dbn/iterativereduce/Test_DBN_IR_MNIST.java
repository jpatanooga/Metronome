package tv.floe.metronome.deeplearning.dbn.iterativereduce;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.irunit.IRUnitDriver;

public class Test_DBN_IR_MNIST {

	@Test
	public void testIR() throws Exception {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/dbn/mnist/app.unit_test.dbn.mnist.properties");
		polr_ir.Setup();

		polr_ir.SimulateRun();

		
		
		
	}
	
}