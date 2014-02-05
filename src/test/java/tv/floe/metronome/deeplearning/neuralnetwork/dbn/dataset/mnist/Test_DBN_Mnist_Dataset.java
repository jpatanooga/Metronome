package tv.floe.metronome.deeplearning.neuralnetwork.dbn.dataset.mnist;

import static org.junit.Assert.*;

import java.io.IOException;

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
		
		PropertyConfigurator.configure( "src/test/resources/log4j/log4j_testing.properties" );
		
		int[] hiddenLayerSizes = { 600, 600, 600 };
		double learningRate = 0.005;
		int preTrainEpochs = 10;
		int fineTuneEpochs = 10;
		int totalNumExamples = 140;
		//int rowLimit = 100;
				
		int batchSize = 20;
		
		// mini-batches through dataset
		MnistDataSetIterator fetcher = new MnistDataSetIterator( batchSize, totalNumExamples );
		DataSet first = fetcher.next();
		int numIns = first.getFirst().numCols();
		int numLabels = first.getSecond().numCols();

		int n_layers = hiddenLayerSizes.length;
		RandomGenerator rng = new MersenneTwister(123);
		
		
		DeepBeliefNetwork dbn = new DeepBeliefNetwork( numIns, hiddenLayerSizes, numLabels, n_layers, rng ); //, Matrix input, Matrix labels);
				
		int recordsProcessed = 0;
		
		do  {
			
			recordsProcessed += batchSize;
			
			System.out.println( "PreTrain: Batch Mode, Processed Total " + recordsProcessed );
			dbn.preTrain( first.getFirst(), 1, learningRate, preTrainEpochs);

			if (fetcher.hasNext()) {
				first = fetcher.next();
			}
			
		} while (fetcher.hasNext());

		fetcher.reset();
		first = fetcher.next();
		
		recordsProcessed = 0;
		
		do {
			
			recordsProcessed += batchSize;
			
			System.out.println( "FineTune: Batch Mode, Processed Total " + recordsProcessed );
			
			
			dbn.finetune( first.getSecond(), learningRate, fineTuneEpochs );
			
			if (fetcher.hasNext()) {
				first = fetcher.next();
			}
			
		} while (fetcher.hasNext());
		
		System.out.println("----------- Training Complete! -----------");
		
		// save model
		
	//	dbn.write( "/tmp/metronome/dbn/TEST_DBN_MNIST/models/mnist.model" );
		
		// now do evaluation of results ....
		fetcher.reset();
		
		ModelTester.evaluateModel(fetcher, dbn);
		
		
	}


}
