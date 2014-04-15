package tv.floe.metronome.deeplearning.neuralnetwork.dbn.dataset.uci.iris;

import static org.junit.Assert.*;

import java.io.FileOutputStream;
import java.io.IOException;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.junit.Test;

import tv.floe.metronome.deeplearning.datasets.DataSet;
import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;
import tv.floe.metronome.deeplearning.dbn.model.evaluation.ModelTester;
import tv.floe.metronome.irunit.IRUnitDriver;
import tv.floe.metronome.math.MatrixUtils;

public class Test_DBN_IrisDataset {

	@Test
	public void testBaseIris() throws IOException {
		
		//PropertyConfigurator.configure( "src/test/resources/log4j/log4j_testing.properties" );
		
		int[] hiddenLayerSizes = { 500, 250, 100 };
		double learningRate = 0.01;
		int preTrainEpochs = 100;
		int fineTuneEpochs = 100;
		int totalNumExamples = 20;
		//int rowLimit = 100;
				
		int batchSize = 1;
		boolean showNetworkStats = true;
		
		// mini-batches through dataset
//		MnistDataSetIterator fetcher = new MnistDataSetIterator( batchSize, totalNumExamples );
//		DataSet first = fetcher.next();
		
		int[] filter = { 0, 1 };
		DataSet recordBatch = null; //this.filterDataset( filter, 20 );
		
		//MatrixUtils.debug_print(recordBatch.getSecond());
		
		//int numIns = first.getFirst().numCols();
		//int numLabels = first.getSecond().numCols();
		
		int numIns = recordBatch.getFirst().numCols();
		int numLabels = recordBatch.getSecond().numCols();

		int n_layers = hiddenLayerSizes.length;
		RandomGenerator rng = new MersenneTwister(123);
		
		
		DeepBeliefNetwork dbn = new DeepBeliefNetwork( numIns, hiddenLayerSizes, numLabels, n_layers, rng ); //, Matrix input, Matrix labels);
				
		dbn.useRegularization = false;
		dbn.setSparsity(0.01);
		dbn.setMomentum(0);
		
		int recordsProcessed = 0;
		
		StopWatch watch = new StopWatch();
		watch.start();
		
		StopWatch batchWatch = new StopWatch();
		
		
//		do  {
			
			recordsProcessed += batchSize;
			
			System.out.println( "PreTrain: Batch Mode, Processed Total " + recordsProcessed + ", Elapsed Time " + watch.toString() );
			
			batchWatch.reset();
			batchWatch.start();
			dbn.preTrain( recordBatch.getFirst(), 1, learningRate, preTrainEpochs);
			batchWatch.stop();
			
			System.out.println( "Batch Training Elapsed Time " + batchWatch.toString() );

			System.out.println( "FineTune: Batch Mode, Processed Total " + recordsProcessed + ", Elapsed Time " + watch.toString() );
			
			
			dbn.finetune( recordBatch.getSecond(), learningRate, fineTuneEpochs );

			watch.stop();
		
		System.out.println("----------- Training Complete! -----------");
		System.out.println( "Processed Total " + recordsProcessed + ", Elapsed Time " + watch.toString() );
		
//		ModelTester.evaluateModel( recordBatch.getFirst(), recordBatch.getSecond(), dbn);
		
		ModelTester.evaluateModel( recordBatch.getFirst(), recordBatch.getSecond(), dbn);
		
	}	
	
	@Test
	public void testLearnDatasetFunctionViaIR() throws Exception {
		
		/*
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/nn/iris/app.unit_test.nn.iris.properties");
		polr_ir.Setup();

		polr_ir.SimulateRun();
*/

		
		
	}

	
	@Test
	public void testLearnDatasetFunctionViaCoreLib() throws Exception {
		
		/*
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/run_profiles/unit_tests/nn/iris/app.unit_test.nn.iris.properties");
		polr_ir.Setup();

		polr_ir.SimulateRun();
*/

		
		
	}
	
	
}
