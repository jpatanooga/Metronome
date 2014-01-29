package tv.floe.metronome.deeplearning.neuralnetwork.dbn.dataset.mnist;

import static org.junit.Assert.*;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;


import tv.floe.metronome.classification.neuralnetworks.iterativereduce.mnist.MNIST_DatasetUtils;
import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;
import tv.floe.metronome.math.MatrixUtils;

public class Test_DBN_Mnist_Dataset {
	
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
		
//		int numIns = first.getFirst().columns;
//		int numLabels = first.getSecond().columns;
		int[] hiddenLayerSizes = { 600, 600, 600 };
		double learningRate = 0.005;
		int preTrainEpochs = 200;
		int fineTuneEpochs = 200;
		int rowLimit = 100;
		
		//MNIST_DatasetUtils dataset_utils = new MNIST_DatasetUtils();
		
		Matrix inputDataset = MNIST_DatasetUtils.getImageDataAsMatrix(rowLimit);
		
		Matrix testSamples = segmentOutSomeTestData( inputDataset, 5 );
		
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
		dbn.finetune( outputLabels, learningRate, fineTuneEpochs );
		
		//for ( int x = 0; x < testSamples.numRows(); x++ ) {
			
		Matrix testPredictedLabels = dbn.predict(testSamples);
		
		System.out.println("\n\n-------------- predictions ------------- ");
		
		//MatrixUtils.debug_print( testPredictedLabels );
		for (int x = 0; x < testPredictedLabels.numRows(); x++ ) {
			System.out.println( x + ": " + testPredictedLabels.viewRow(x).maxValueIndex() );
		}
			
		System.out.println("\n\n-------------- actual ------------- ");
		
//		MatrixUtils.debug_print( outputLabels );
		for (int x = 0; x < outputLabels.numRows(); x++ ) {
			System.out.println( x + ": " + outputLabels.viewRow(x).maxValueIndex() );
		}

		
		//}
		
		
		
		
	}

}
