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

		
		WorkerNode singleWorker = (WorkerNode)polr_ir.getWorker().get(0);
	//	WorkerNode secondWorker = (WorkerNode)polr_ir.getWorker().get(1);
		
		MasterNode master = (MasterNode) polr_ir.getMaster();
		
		System.out.println("\n\nComplete: ");
		//Utils.PrintVector( master.polr.getBeta().viewRow(0) );

//		this.scoreNeuralNetworkXor( master.first_worker_copy );
		
		System.out.println("Worker 1 ");
		//TestTwoWorkersXOR_IR_NN.scoreNeuralNetworkXor( singleWorker.nn );
		//Utils.PrintNeuralNetwork( singleWorker.nn );
		System.out.println("w1 > RMSE: " + singleWorker.lastRMSE );
		
	}
}
