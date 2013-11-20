package tv.floe.metronome.classification.neuralnetworks.iterativereduce.uci.dermatology;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.iterativereduce.MasterNode;
import tv.floe.metronome.classification.neuralnetworks.iterativereduce.WorkerNode;
import tv.floe.metronome.irunit.IRUnitDriver;

public class Test_UCI_Dermatology_NN_IR {

	@Test
	public void testLearnFunctionViaIRNN_MLP() throws Exception {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/nn/dermatology/app.unit_test.nn.dermatology.properties");
		polr_ir.Setup();

		polr_ir.SimulateRun();

		
	}
}
