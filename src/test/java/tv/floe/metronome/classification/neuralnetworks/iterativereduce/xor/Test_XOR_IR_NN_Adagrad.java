package tv.floe.metronome.classification.neuralnetworks.iterativereduce.xor;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.iterativereduce.MasterNode;
import tv.floe.metronome.irunit.IRUnitDriver;

public class Test_XOR_IR_NN_Adagrad {

	@Test
	public void testLearnXORFunctionViaIRNN_MLP() throws Exception {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/nn/xor/app.unit_test.nn.xor.singlenode.adagrad.properties");
		polr_ir.Setup();

		polr_ir.SimulateRun();

		MasterNode master = (MasterNode) polr_ir.getMaster();
		
		System.out.println("\n\nComplete: ");

		XOR_DatasetTestUtils.scoreNeuralNetworkXor( master.master_nn );
		
	}
}
