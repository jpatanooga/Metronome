package tv.floe.metronome.classification.neuralnetworks.iterativereduce.mnist;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.iterativereduce.MasterNode;
import tv.floe.metronome.classification.neuralnetworks.iterativereduce.WorkerNode;
import tv.floe.metronome.irunit.IRUnitDriver;

public class Test_MNISTDataset_IR_NN {

	@Test
	public void testLearnIrisFunctionViaIRNN_MLP() throws Exception {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/nn/mnist/app.unit_test.nn.mnist.properties");
		polr_ir.Setup();

		polr_ir.SimulateRun();

		
		
		
	}
	
}
