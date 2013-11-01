package tv.floe.metronome.classification.neuralnetworks.learning;

import static org.junit.Assert.*;

import java.io.BufferedReader;
import java.io.FileReader;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import tv.floe.metronome.classification.neuralnetworks.conf.Config;
import tv.floe.metronome.classification.neuralnetworks.networks.MultiLayerPerceptronNetwork;
import tv.floe.metronome.classification.neuralnetworks.utils.Utils;

public class TestLeastMeanSquaresLearningAlgorithm {

	@Test
	public void testTrain() throws Exception {

		Config c = new Config();
		c.parse(null); // default layer: 2-3-2
		c.setConfValue("useBiasNeuron", "true");
		
	
		int[] neurons = { 4, 16, 3 };
		c.setLayerNeuronCounts( neurons );
		
		// layers: 4, 16, 3
        MultiLayerPerceptronNetwork neuralNet = new MultiLayerPerceptronNetwork();
        neuralNet.buildFromConf(c);
		
        
        
        // now test the lms_algo
        
		LeastMeanSquaresLearningAlgorithm lms_algo = new LeastMeanSquaresLearningAlgorithm();
		lms_algo.setNeuralNetwork(neuralNet);
		
		int num_inputs = 4;
		int num_outputs = 3;
		
		BufferedReader reader = new BufferedReader( new FileReader("src/test/resources/data/iris/iris_data_normalised.txt") );
		
		
		String line = reader.readLine();
		

		if (null == line || line.trim().equals("")) {
			
			System.out.println("> bad line > " + line );
			
		} else {
				

			Vector vec_inputs = new RandomAccessSparseVector( num_inputs );
			
			Vector vec_outputs = new RandomAccessSparseVector( num_outputs );
			
			Utils.parseCSVRecord(vec_inputs, num_inputs, vec_outputs, num_outputs, line);
			
			assertEquals( num_inputs, vec_inputs.size() );
			
			assertEquals( num_outputs, vec_outputs.size() );
			
			//neuralNet.train(vec_outputs, vec_inputs);
			
			lms_algo.train(vec_outputs, vec_inputs);

		}
		
		
		
	}
	
	@Test
	public void testUpdateNetworkWeights() {
		
		
	}
	
	@Test
	public void testUpdateNeuronWeights() {
		
		
	}

	@Test
	public void testHasReachedStoppingCondition() {
		
		
	}
	
	@Test
	public void testErrorChangeStalled() {
		
	}
	
	@Test
	public void testCalculateOutputError() {
		
		String rec_0 = "0.64556962, 0.795454545, 0.202898551, 0.08, 1, 0, 0";
		
		String rec_0_bad_answer_0 = "0.64556962, 0.795454545, 0.202898551, 0.08, 0, 0, 0";
		String rec_0_bad_answer_1 = "0.64556962, 0.795454545, 0.202898551, 0.08, 0, 1, 0";
		String rec_0_bad_answer_2 = "0.64556962, 0.795454545, 0.202898551, 0.08, 0.5, 0, 0";
		String rec_0_bad_answer_3 = "0.64556962, 0.795454545, 0.202898551, 0.08, 0, 0, 1";
		
		String rec_1 = "0.620253165, 0.681818182, 0.202898551, 0.08, 1, 0, 0";

		int num_inputs = 4;
		int num_outputs = 3;
		
		
		Vector vec_inputs = new RandomAccessSparseVector( num_inputs );
		
		Vector vec_outputs = new RandomAccessSparseVector( num_outputs );
		Vector vec_outputs_bad_answer_0 = new RandomAccessSparseVector( num_outputs );
		Vector vec_outputs_bad_answer_1 = new RandomAccessSparseVector( num_outputs );
		Vector vec_outputs_bad_answer_2 = new RandomAccessSparseVector( num_outputs );
		Vector vec_outputs_bad_answer_3 = new RandomAccessSparseVector( num_outputs );
		
		Utils.parseCSVRecord(vec_inputs, num_inputs, vec_outputs, num_outputs, rec_0);
		Utils.parseCSVRecord(vec_inputs, num_inputs, vec_outputs_bad_answer_0, num_outputs, rec_0_bad_answer_0);
		Utils.parseCSVRecord(vec_inputs, num_inputs, vec_outputs_bad_answer_1, num_outputs, rec_0_bad_answer_1);
		Utils.parseCSVRecord(vec_inputs, num_inputs, vec_outputs_bad_answer_2, num_outputs, rec_0_bad_answer_2);
		Utils.parseCSVRecord(vec_inputs, num_inputs, vec_outputs_bad_answer_3, num_outputs, rec_0_bad_answer_3);
		
		
		
		LeastMeanSquaresLearningAlgorithm lms_algo = new LeastMeanSquaresLearningAlgorithm();

		// no error here
		double[] err_0 = lms_algo.calculateOutputError(vec_outputs, vec_outputs);
		assertEquals( 0.0, err_0[0], 0.0000d );
		assertEquals( 0.0, err_0[1], 0.0000d );
		assertEquals( 0.0, err_0[2], 0.0000d );
		
		

		double[] err_1 = lms_algo.calculateOutputError(vec_outputs, vec_outputs_bad_answer_0);
		assertEquals( 1.0, err_1[0], 0.0000d );
		assertEquals( 0.0, err_1[1], 0.0000d );
		assertEquals( 0.0, err_1[2], 0.0000d );

		double[] err_2 = lms_algo.calculateOutputError(vec_outputs, vec_outputs_bad_answer_1);
		assertEquals( 1.0, err_2[0], 0.0000d );
		assertEquals( -1.0, err_2[1], 0.0000d );
		assertEquals( 0.0, err_2[2], 0.0000d );

		// 0.5, 0, 0
		double[] err_3 = lms_algo.calculateOutputError(vec_outputs, vec_outputs_bad_answer_2);
		assertEquals( 0.5, err_3[0], 0.0000d );
		assertEquals( 0.0, err_3[1], 0.0000d );
		assertEquals( 0.0, err_3[2], 0.0000d );
		
		// 0, 0, 1
		double[] err_4 = lms_algo.calculateOutputError(vec_outputs, vec_outputs_bad_answer_3);
		assertEquals( 1.0, err_4[0], 0.0000d );
		assertEquals( 0.0, err_4[1], 0.0000d );
		assertEquals( -1.0, err_4[2], 0.0000d );
		
		
		
	}
	
	@Test
	public void testAddToSquaredErrorSum() {
		
		LeastMeanSquaresLearningAlgorithm lms_algo = new LeastMeanSquaresLearningAlgorithm();

		double[] outputError_0 = { 0.1d, 0.1d, 0.1d };
		double[] outputError_1 = { 0.1d, 0.2d, 0.3d };
		
		lms_algo.addToSquaredErrorSum(outputError_0);
		
		assertEquals( 0.03d, lms_algo.getTotalSquaredError(), 0.000001d );

		lms_algo.clearTotalSquaredError();
		
		lms_algo.addToSquaredErrorSum(outputError_1);
		
		assertEquals( 0.14d, lms_algo.getTotalSquaredError(), 0.000001d );
		
		
		// now make sure RMSE works correctly
		lms_algo.clearTotalSquaredError();
		lms_algo.addToSquaredErrorSum(outputError_0);
		lms_algo.addToSquaredErrorSum(outputError_1);
		
		lms_algo.setRecordsSeen_Debug(2);
		
		double rmse = lms_algo.calcRMSError();
		
		double rmse_answer = Math.sqrt((0.03d +  0.14d) / 2.0);
		
		assertEquals(rmse_answer, rmse, 0.0);
		
	}
	
	
	
	
	
}
