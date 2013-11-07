package tv.floe.metronome.classification.neuralnetworks.iterativereduce.bcw;

import static org.junit.Assert.*;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.iterativereduce.MasterNode;
import tv.floe.metronome.classification.neuralnetworks.iterativereduce.WorkerNode;
import tv.floe.metronome.io.records.MetronomeRecordFactory;
import tv.floe.metronome.irunit.IRUnitDriver;

public class Test_BreastCancerWisconsin_IR_NN {

	@Test
	public void testVectorizeBCWLine() {
		
		String line = "0:1.0 1:0.7 2:0.7 3:0.6 4:0.4 5:1.0 6:0.4 7:0.1 8:0.2 | 0:1.0";
		
		String schema = "i:9 | o:1";
		
		MetronomeRecordFactory rec_factory = new MetronomeRecordFactory(schema);
		
		assertEquals(rec_factory.getInputVectorSize(), 9);
		assertEquals(rec_factory.getOutputVectorSize(), 1);
		
		Vector v_in_0 = new RandomAccessSparseVector(rec_factory.getInputVectorSize());
		Vector v_out_0 = new RandomAccessSparseVector(rec_factory.getOutputVectorSize());

		rec_factory.vectorizeLine( line, v_in_0, v_out_0 );
		
		assertEquals(0.7, v_in_0.get(1), 0.0);
		assertEquals(0.7, v_in_0.get(2), 0.0);
		
		// test output
		assertEquals(1.0, v_out_0.get(0), 0.0);
		//assertEquals(0.0, v_out_0.get(1), 0.0);
		//assertEquals(0.0, v_out_0.get(2), 0.0);		
		
	}
	
	@Test
	public void testLearnBCWFunctionViaIRNN_MLP() throws Exception {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/nn/breast_cancer_wisconsin/app.unit_test.nn.bcw.properties");
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
