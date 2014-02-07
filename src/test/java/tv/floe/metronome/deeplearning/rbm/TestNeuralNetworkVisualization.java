package tv.floe.metronome.deeplearning.rbm;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

import tv.floe.metronome.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import tv.floe.metronome.deeplearning.neuralnetwork.core.NeuralNetworkGradient;
import tv.floe.metronome.deeplearning.rbm.visualization.NeuralNetworkVisualizer;


public class TestNeuralNetworkVisualization {


	@Test
	public void testPlot() throws IOException {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(20,20);
		MersenneTwister rand = new MersenneTwister(123);
/*
		RestrictedBoltzmannMachine da = new RBM.Builder().numberOfVisible(784).numHidden(400).withRandom(rand).renderWeights(100)
				.useRegularization(false)
				.withMomentum(0).build();
*/
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine( 784, 400, null );

		
		Matrix input = fetcher.next().getFirst();
		rbm.trainingDataset = input;

		NeuralNetworkVisualizer plotter = new NeuralNetworkVisualizer();
		NeuralNetworkGradient gradient = rbm.getGradient(new Object[]{1,0.01});

		for(int i = 0; i < 1000; i++) {
			
			rbm.trainTillConvergence(0.01, 1, input);
			
		}




	}
	
	
}
