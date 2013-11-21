package tv.floe.metronome.classification.neuralnetworks.iterativereduce.nist.handwriting;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.iterativereduce.MasterNode;
import tv.floe.metronome.classification.neuralnetworks.iterativereduce.WorkerNode;
import tv.floe.metronome.irunit.IRUnitDriver;

public class Test_NIST_HandWritingDataset_IR_NN {

	@Test
	public void testLearnNISTHandwritingFunctionViaIRNN_MLP() throws Exception {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/nn/nist/handwritten_digits/app.unit_test.nn.nist.handwritten.0.properties");
		polr_ir.Setup();

		polr_ir.SimulateRun();

		
	}
}
