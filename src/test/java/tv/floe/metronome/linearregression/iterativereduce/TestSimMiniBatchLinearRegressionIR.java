package tv.floe.metronome.linearregression.iterativereduce;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.irunit.IRUnitDriver;
import tv.floe.metronome.utils.Utils;

public class TestSimMiniBatchLinearRegressionIR {

	@Test
	public void test() {
//		fail("Not yet implemented");
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/app.unit_test_linearReg_miniBatch.properties");
		polr_ir.Setup();
		polr_ir.SimulateRun();
		
		MasterNode master = (MasterNode) polr_ir.getMaster();
		
		System.out.println("\n\nComplete: ");
		Utils.PrintVector( master.polr.getBeta().viewRow(0) );
		
	}

}
