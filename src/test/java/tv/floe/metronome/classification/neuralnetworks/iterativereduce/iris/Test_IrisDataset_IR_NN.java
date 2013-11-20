package tv.floe.metronome.classification.neuralnetworks.iterativereduce.iris;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.iterativereduce.MasterNode;
import tv.floe.metronome.classification.neuralnetworks.iterativereduce.WorkerNode;
import tv.floe.metronome.classification.neuralnetworks.iterativereduce.xor.TestTwoWorkersXOR_IR_NN;
import tv.floe.metronome.classification.neuralnetworks.utils.Utils;
import tv.floe.metronome.irunit.IRUnitDriver;

public class Test_IrisDataset_IR_NN {

	@Test
	public void testLearnIrisFunctionViaIRNN_MLP() throws Exception {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/nn/iris/app.unit_test.nn.iris.properties");
		polr_ir.Setup();

		polr_ir.SimulateRun();


		
		
	}
		
}
