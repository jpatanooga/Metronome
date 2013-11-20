package tv.floe.metronome.classification.neuralnetworks.iterativereduce.abalone;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.iterativereduce.MasterNode;
import tv.floe.metronome.classification.neuralnetworks.iterativereduce.WorkerNode;
import tv.floe.metronome.irunit.IRUnitDriver;

public class Test_Abalone_NN_IR {

	@Test
	public void testLearnFunctionViaIRNN_MLP() throws Exception {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/nn/abalone/app.unit_test.nn.abalone.properties");
		polr_ir.Setup();

		polr_ir.SimulateRun();

		
	}
}
