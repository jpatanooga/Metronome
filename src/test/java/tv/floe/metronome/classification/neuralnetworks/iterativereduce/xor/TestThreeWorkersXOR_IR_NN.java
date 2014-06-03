package tv.floe.metronome.classification.neuralnetworks.iterativereduce.xor;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.iterativereduce.MasterNode;
import tv.floe.metronome.classification.neuralnetworks.iterativereduce.WorkerNode;
import tv.floe.metronome.irunit.IRUnitDriver;

public class TestThreeWorkersXOR_IR_NN {

	@Test
	public void testLearnXORFunctionViaIRNN_MLP() throws Exception {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/nn/xor/app.unit_test.nn.xor.threeworkers.properties");
		polr_ir.Setup();

		polr_ir.SimulateRun();

		
//		WorkerNode singleWorker = (WorkerNode)polr_ir.getWorker().get(0);
//		WorkerNode secondWorker = (WorkerNode)polr_ir.getWorker().get(1);
		
		MasterNode master = (MasterNode) polr_ir.getMaster();
		
		System.out.println("\n\nComplete: ");

//		TestTwoWorkersXOR_IR_NN.scoreNeuralNetworkXor( master.master_nn );
		
		
		
	}
}
