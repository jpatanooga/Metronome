package tv.floe.metronome.deeplearning.rbm.datasets.mnist;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.mahout.math.Matrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tv.floe.metronome.deeplearning.datasets.DataSet;
import tv.floe.metronome.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import tv.floe.metronome.deeplearning.rbm.RestrictedBoltzmannMachine;
import tv.floe.metronome.deeplearning.rbm.visualization.DrawMnistGreyscale;
import tv.floe.metronome.math.MatrixUtils;


public class TestMNIST_On_RBMs {

	private static Logger log = LoggerFactory.getLogger(TestMNIST_On_RBMs.class);

	@Test
	public void testMnist() throws Exception {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(100,100);
		MersenneTwister rand = new MersenneTwister(123);

		
		DataSet first = fetcher.next();
/*
		RestrictedBoltzmannMachine da = new RBM.Builder().numberOfVisible(784).numHidden(400).withRandom(rand).renderWeights(1000)
				.useRegularization(false)
				.withMomentum(0).build();
*/
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine( 784, 400, null );

		// TODO: investigate "render weights"



		rbm.trainingDataset = first.getFirst();

		rbm.trainTillConvergence( first.getFirst(), 0.01, new Object[]{ 1 } );
		Matrix reconstruct = rbm.reconstruct(first.getFirst());

		for (int j = 0; j < first.numExamples(); j++) {
			
			Matrix draw1 = first.get(j).getFirst().times(255);
			
			//Matrix reconstructed2 = reconstruct.getRow(j);
			Matrix reconstructed2 = MatrixUtils.viewRowAsMatrix(reconstruct, j);
			
			//Matrix draw2 = MatrixUtils.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);
			Matrix draw2 = MatrixUtils.genBinomialDistribution(reconstructed2,1,new MersenneTwister(123)).times(255);

			DrawMnistGreyscale d = new DrawMnistGreyscale(draw1);
			d.title = "REAL";
			d.draw();
			d.frame.setLocation(100, 200);
			
			DrawMnistGreyscale d2 = new DrawMnistGreyscale( draw2, 100, 100 );
			d2.title = "TEST";
			d2.draw();
			d2.frame.setLocation(300, 200);

			Thread.sleep(1000);
			d.frame.dispose();
			d2.frame.dispose();
			
			System.out.println("Viz Example: " + j);

		}
		
		for (int i = 0; i < 3000; i++) {
			
			if (i% 500 == 0 || i == 0) {
				
				reconstruct = rbm.reconstruct(first.getFirst());
				
				if (i > 0)
					
					for (int j = 0; j < first.numExamples(); j++) {
						

						
						Matrix draw1 = first.get(j).getFirst().times(255);
						Matrix reconstructed2 = MatrixUtils.viewRowAsMatrix(reconstruct, j);
						Matrix draw2 = MatrixUtils.genBinomialDistribution(reconstructed2,1,new MersenneTwister(123)).times(255);
						
						
						DrawMnistGreyscale d = new DrawMnistGreyscale(draw1);
						d.title = "REAL";
						d.draw();
						DrawMnistGreyscale d2 = new DrawMnistGreyscale(draw2,100,100);
						d2.title = "TEST";
						d2.draw();
						Thread.sleep(1000);
						d.frame.dispose();
						d2.frame.dispose();

					}
			}
			
			rbm.train(first.getFirst(), 0.01, new Object[]{1});
			
			log.info("Negative log likelihood " + rbm.getReConstructionCrossEntropy());


		}







	}

}
