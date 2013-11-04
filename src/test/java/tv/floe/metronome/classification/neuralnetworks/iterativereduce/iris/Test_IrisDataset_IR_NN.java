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
	public void testLearnXORFunctionViaIRNN_MLP() throws Exception {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/nn/iris/app.unit_test.nn.iris.properties");
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
		/*
		System.out.println("Worker 2 ");
		TestTwoWorkersXOR_IR_NN.scoreNeuralNetworkXor( secondWorker.nn );
		Utils.PrintNeuralNetwork( secondWorker.nn );
		*/
		
		//System.out.println("Gobal Results");
		//TestTwoWorkersXOR_IR_NN.scoreNeuralNetworkXor( master.master_nn );
		//Utils.PrintNeuralNetwork(  master.master_nn );
		
	}
		
}
