package tv.floe.metronome.deeplearning.neuralnetwork.dbn.dataset.mnist;

import static org.junit.Assert.*;

import org.apache.mahout.math.Matrix;
import org.junit.Test;

import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;

public class Test_DBN_Mnist_Dataset {
	
	
	public static Matrix generateMNISTDataset() {
		
		
		
		return null;
		
	}

	/**
	 * For each hidden / RBM layer, the visible units are dictated by the number of incoming
	 * entries in the input matrix
	 * 
	 * The hidden units are manually set by us here 
	 * 
	 */
	@Test
	public void testMnist() {
		
//		int numIns = first.getFirst().columns;
//		int numLabels = first.getSecond().columns;
		int[] hiddenLayerSizes = { 1000, 1000, 2000 };
		double learningRate = 0.1;
		
		Matrix inputDataset = generateMNISTDataset();
		
		int n_ins = 0; // number of elements in input vector
		int n_outs = 10; // 0 - 9
		int n_layers = hiddenLayerSizes.length;
		

		DeepBeliefNetwork dbn = new DeepBeliefNetwork();
		
//		dbn.pretrain(first.getFirst(),1, lr, 50);
//		dbn.finetune(first.getSecond(),lr, 50);
		
		
		
	}

}
