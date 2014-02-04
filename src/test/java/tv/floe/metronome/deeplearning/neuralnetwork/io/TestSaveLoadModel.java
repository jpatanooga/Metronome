package tv.floe.metronome.deeplearning.neuralnetwork.io;

import static org.junit.Assert.*;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.junit.Test;

import tv.floe.metronome.deeplearning.datasets.DataSet;
import tv.floe.metronome.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;

public class TestSaveLoadModel {

	@Test
	public void testLoadSaveModel() throws IOException {

	
		
		int[] hiddenLayerSizes = { 600, 600, 600 };
		double learningRate = 0.005;
		int preTrainEpochs = 5;
		int fineTuneEpochs = 5;
		int totalNumExamples = 50;
		//int rowLimit = 100;
				
		int batchSize = 10;
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
		
		System.out.println("----------- Saving Model -----------");
		dbn.write( "/tmp/metronome/dbn/TEST_DBN_MNIST/models/mnist.model" );
		
		// now do evaluation of results ....
			
			
	
	
	}

}
