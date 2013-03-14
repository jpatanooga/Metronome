package tv.floe.metronome.linearregression.iterativereduce;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.irunit.IRUnitDriver;
import tv.floe.metronome.utils.Utils;

/**
 * TODO: build a re-usable test harness to simulate the workings of a IterativeReduce run
 * 
 * 
 * @author josh
 *
 */
public class TestSimulateLinearRegressionIterativeReduce {

	@Test
	public void test() {
		//fail("Not yet implemented");
		//IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/app.unit_test.SATScores_3splits.properties");
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/app.unit_test.Rsynth.2_10_var2.properties");
		polr_ir.Setup();

		polr_ir.SimulateRun();
		
		MasterNode master = (MasterNode) polr_ir.getMaster();
		
		System.out.println("\n\nComplete: ");
		Utils.PrintVector( master.polr.getBeta().viewRow(0) );
		
	}

}
