package tv.floe.metronome.deeplearning.neuralnetwork.dbn.dataset.mnist;

import static org.junit.Assert.*;

import java.io.FileOutputStream;
import java.io.IOException;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.log4j.PropertyConfigurator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;





import tv.floe.metronome.classification.neuralnetworks.iterativereduce.mnist.MNIST_DatasetUtils;
import tv.floe.metronome.deeplearning.datasets.DataSet;
import tv.floe.metronome.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;
import tv.floe.metronome.deeplearning.dbn.model.evaluation.ModelTester;
import tv.floe.metronome.eval.Evaluation;
import tv.floe.metronome.math.MatrixUtils;

public class Test_DBN_Mnist_Dataset {
	
	private static Logger log = LoggerFactory.getLogger(Test_DBN_Mnist_Dataset.class);
	
	
	public static Matrix segmentOutSomeTestData(Matrix input, int max_count) {
		
		int rows = max_count;
		
		if (max_count > input.numRows()) {
			
			rows = input.numRows();
			
		}
		
		Matrix samples = new DenseMatrix( rows, input.numCols() );
		
		for (int x = 0; x < rows; x++ ) {
			
			samples.assignRow(x, input.viewRow(x) );
			
		}
		
		
		return samples;
		
		
	}
	/*
	@Test
	public void testMeh() throws InterruptedException {
		
		org.apache.commons.lang3.time.StopWatch foo = new org.apache.commons.lang3.time.StopWatch();
		
		foo.start();
		Thread.sleep(5000);
		//foo.stop();
		
		System.out.println( foo.toString() );
		
		
	}
	*/
	

	
	
	/**
	 * For each hidden / RBM layer, the visible units are dictated by the number of incoming
	 * entries in the input matrix
	 * 
	 * The hidden units are manually set by us here 
	 * 
	 * TODO: 
	 * - 1. generate MNIST input data as a matrix
	 * 
	 * - 2. train DBN
	 * 
	 * - 3. generate number correct
	 * @throws IOException 
	 * 
	 */
	@Test
	public void testMnist() throws IOException {
		
		//PropertyConfigurator.configure( "src/test/resources/log4j/log4j_testing.properties" );
		
		int[] hiddenLayerSizes = { 500, 250, 100 };
		double learningRate = 0.001;
		int preTrainEpochs = 1000;
		int fineTuneEpochs = 1000;
		int totalNumExamples = 10;
		//int rowLimit = 100;
				
		int batchSize = 10;
		boolean showNetworkStats = true;
		
		// mini-batches through dataset
		MnistDataSetIterator fetcher = new MnistDataSetIterator( batchSize, totalNumExamples );
		DataSet first = fetcher.next();
		int numIns = first.getFirst().numCols();
		int numLabels = first.getSecond().numCols();

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
		
		
		do  {
			
			recordsProcessed += batchSize;
			
			System.out.println( "PreTrain: Batch Mode, Processed Total " + recordsProcessed + ", Elapsed Time " + watch.toString() );
			
			batchWatch.reset();
			batchWatch.start();
			dbn.preTrain( first.getFirst(), 1, learningRate, preTrainEpochs);
			batchWatch.stop();
			
			System.out.println( "Batch Training Elapsed Time " + batchWatch.toString() );

			//System.out.println( "DBN Network Stats:\n" + dbn.generateNetworkSizeReport() );

			
			if (fetcher.hasNext()) {
				first = fetcher.next();
			}
			
		} while (fetcher.hasNext());

		fetcher.reset();
		first = fetcher.next();
		
		recordsProcessed = 0;
		
		do {
			
			recordsProcessed += batchSize;
			
			System.out.println( "FineTune: Batch Mode, Processed Total " + recordsProcessed + ", Elapsed Time " + watch.toString() );
			
			
			dbn.finetune( first.getSecond(), learningRate, fineTuneEpochs );
			
			if (fetcher.hasNext()) {
				first = fetcher.next();
			}
			
		} while (fetcher.hasNext());
		
		watch.stop();
		
		System.out.println("----------- Training Complete! -----------");
		System.out.println( "Processed Total " + recordsProcessed + ", Elapsed Time " + watch.toString() );
		
		// save model
		
	//	dbn.write( "/tmp/metronome/dbn/TEST_DBN_MNIST/models/mnist.model" );
		
		FileOutputStream oFileOutStream = new FileOutputStream( "/tmp/Metronome_DBN_Mnist.model", false);
		dbn.write( oFileOutStream );
		
		
		// now do evaluation of results ....
		fetcher.reset();
		
		ModelTester.evaluateModel(fetcher, dbn);
		
		
	}

	/**
	 * Note: not meant as a real unit test
	 * 
	 * Meant to be used to collect perf stats
	 * 
	 * @throws IOException
	 */
	@Test
	public void testMnist_SingleBatchAvgPreTrainTime() throws IOException {
		
		//PropertyConfigurator.configure( "src/test/resources/log4j/log4j_testing.properties" );
		
		int[] hiddenLayerSizes = { 500, 500, 500 };
		double learningRate = 0.01;
		int preTrainEpochs = 100;
		int fineTuneEpochs = 100;
		int totalNumExamples = 1000;
		//int rowLimit = 100;
				
		int batchSize = 50;
		boolean showNetworkStats = true;
		
		// mini-batches through dataset
		MnistDataSetIterator fetcher = new MnistDataSetIterator( batchSize, totalNumExamples );
		DataSet first = fetcher.next();
		int numIns = first.getFirst().numCols();
		int numLabels = first.getSecond().numCols();

		int n_layers = hiddenLayerSizes.length;
		RandomGenerator rng = new MersenneTwister(123);
		
		
		DeepBeliefNetwork dbn = new DeepBeliefNetwork( numIns, hiddenLayerSizes, numLabels, n_layers, rng ); //, Matrix input, Matrix labels);
				
		dbn.useRegularization = false;
		dbn.setSparsity(0.01);
		dbn.setMomentum(0);
		
		
		int recordsProcessed = 0;
		int batchesProcessed = 0;
		long totalBatchProcessingTime = 0;
		
		StopWatch watch = new StopWatch();
		watch.start();
		
		StopWatch batchWatch = new StopWatch();
		
		
		do  {
			
			recordsProcessed += batchSize;
			batchesProcessed++;
			
			System.out.println( "PreTrain: Batch Mode, Processed Total " + recordsProcessed + ", Elapsed Time " + watch.toString() );
			
			batchWatch.reset();
			batchWatch.start();
			dbn.preTrain( first.getFirst(), 1, learningRate, preTrainEpochs);
			batchWatch.stop();
			
			totalBatchProcessingTime += batchWatch.getTime();
			
			System.out.println( "Batch Training Elapsed Time " + batchWatch.toString() );

			//System.out.println( "DBN Network Stats:\n" + dbn.generateNetworkSizeReport() );

			
			if (fetcher.hasNext()) {
				first = fetcher.next();
			}
			
		} while (fetcher.hasNext());

		double avgBatchTime = totalBatchProcessingTime / batchesProcessed;
		double avgBatchSeconds = avgBatchTime / 1000;
		double avgBatchMinutes = avgBatchSeconds / 60;
		
		System.out.println("--------------------------");
		System.out.println("Avg Batch Processing Time: " + avgBatchMinutes + " minutes per batches of " + batchSize);
		System.out.println("--------------------------");
		
	}	
	
	

}
