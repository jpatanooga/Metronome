package tv.floe.metronome.deeplearning.rbm;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

import tv.floe.metronome.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import tv.floe.metronome.deeplearning.neuralnetwork.gradient.NeuralNetworkGradient;
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
		rbm.useRegularization = false;
		rbm.momentum = 0 ;
		rbm.sparsity = 0.01;

		double learningRate = 0.001;
		
		Matrix input = fetcher.next().getFirst();
		rbm.trainingDataset = input;

		NeuralNetworkVisualizer plotter = new NeuralNetworkVisualizer();
//		NeuralNetworkGradient gradient = rbm.getGradient(new Object[]{1,0.01});

		for(int i = 0; i < 1; i++) {
			
			//rbm.trainTillConvergence(0.01, 1, input);
			rbm.trainTillConvergence(input, learningRate, new Object[]{ 1, learningRate, 2 } );
			System.out.println( "> Epoch > " + i );
			
		}
		
		System.out.println(" ----- done ----- \nPlotting Weights:");
		
		plotter.plotWeights(rbm);




	}
	
	
}
