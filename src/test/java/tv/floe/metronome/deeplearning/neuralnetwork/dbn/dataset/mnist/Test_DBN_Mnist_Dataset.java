package tv.floe.metronome.deeplearning.neuralnetwork.dbn.dataset.mnist;

import static org.junit.Assert.*;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;
import org.junit.Test;


import tv.floe.metronome.classification.neuralnetworks.iterativereduce.mnist.MNIST_DatasetUtils;
import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;
import tv.floe.metronome.math.MatrixUtils;

public class Test_DBN_Mnist_Dataset {
	

	
	
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
		
//		int numIns = first.getFirst().columns;
//		int numLabels = first.getSecond().columns;
		int[] hiddenLayerSizes = { 1000, 1000, 2000 };
		double learningRate = 0.1;
		int preTrainEpochs = 2;
		int fineTuneEpochs = 2;
		int rowLimit = 10;
		
		//MNIST_DatasetUtils dataset_utils = new MNIST_DatasetUtils();
		
		Matrix inputDataset = MNIST_DatasetUtils.getImageDataAsMatrix(rowLimit);
		
		Matrix outputLabels = MNIST_DatasetUtils.getLabelsAsMatrix(rowLimit);
		
		int n_ins = inputDataset.numCols(); // number of elements in input vector 
		int n_outs = 10; // 0 - 9 == number of classes of labels
		int n_layers = hiddenLayerSizes.length;
		RandomGenerator rng = new MersenneTwister(123);
		
		System.out.println( "input records: " + inputDataset.numRows() );
		System.out.println( "input labels: " + outputLabels.numRows() );

		assertEquals( rowLimit, inputDataset.numRows() );
		assertEquals( 784, inputDataset.numCols() );

		assertEquals( rowLimit, outputLabels.numRows() );
		assertEquals( 10, outputLabels.numCols() );
		
//		MatrixUtils.debug_print_row(inputDataset, 0);
//		MatrixUtils.debug_print_row(inputDataset, 10);
		
		
		DeepBeliefNetwork dbn = new DeepBeliefNetwork(n_ins, hiddenLayerSizes, n_outs, n_layers, rng ); //, Matrix input, Matrix labels);
		
		dbn.preTrain( inputDataset, 1, learningRate, preTrainEpochs );
//		dbn.finetune( outputLabels, learningRate, fineTuneEpochs );
		
		
		
		
	}

}
