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

	
	private void renderExample( Matrix draw1, Matrix reconstructed2, Matrix draw2 ) throws InterruptedException {
		
/*
		Matrix draw1 = first.get(j).getFirst().times(255);
		
		//Matrix reconstructed2 = reconstruct.getRow(j);
		Matrix reconstructed2 = MatrixUtils.viewRowAsMatrix(reconstruct, j);
		
		//Matrix draw2 = MatrixUtils.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);
		Matrix draw2 = MatrixUtils.genBinomialDistribution(reconstructed2,1,new MersenneTwister(123)).times(255);
*/
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
		
		
		
	}
	
	@Test
	public void testMnist() throws Exception {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(100,100);
		MersenneTwister rand = new MersenneTwister(123);

		double learningRate = 0.001;
		
		DataSet first = fetcher.next();
/*
		RestrictedBoltzmannMachine da = new RBM.Builder().numberOfVisible(784).numHidden(400).withRandom(rand).renderWeights(1000)
				.useRegularization(false)
				.withMomentum(0).build();
*/
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine( 784, 400, null );

		// TODO: investigate "render weights"



		rbm.trainingDataset = first.getFirst();

		System.out.println(" ----- Training ------");
		
		for(int i = 0; i < 10; i++) {
			
			System.out.println("Epoch " + i + " Negative Log Likelhood: " + rbm.getReConstructionCrossEntropy() );
			
			//rbm.trainTillConvergence( first.getFirst(), learningRate, new Object[]{ 1 } );
			rbm.trainTillConvergence(learningRate, 1, first.getFirst());
			
			
		}
		
		Matrix reconstruct = rbm.reconstruct(first.getFirst());

		log.info("Negative log likelihood " + rbm.getReConstructionCrossEntropy());

		System.out.println(" ----- Visualizing Reconstructions ------");
		
		for (int j = 0; j < 10; j++) {
			
			// get the actual image we're looking at
			Matrix draw1 = first.get(j).getFirst().times(255);
			
			// get the reconstruction row that matches this image
			Matrix reconstructed2 = MatrixUtils.viewRowAsMatrix(reconstruct, j);
			
			// now generate a new image based on the reconstruction probabilities
			Matrix draw2 = MatrixUtils.genBinomialDistribution(reconstructed2,1,new MersenneTwister(123)).times(255);

			// draw the original image
			DrawMnistGreyscale d = new DrawMnistGreyscale(draw1);
			d.title = "REAL";
			d.draw();
			d.frame.setLocation(100, 200);
			
			// now draw the reconstructed image
			DrawMnistGreyscale d2 = new DrawMnistGreyscale( draw2, 100, 100 );
			d2.title = "TEST";
			d2.draw();
			d2.frame.setLocation(300, 200);

			Thread.sleep(5000);
			d.frame.dispose();
			d2.frame.dispose();
			
			System.out.println("Viz Example: " + j);

		}
		
		
		for (int i = 0; i < 3000; i++) {
			
			if (i% 100 == 0 || i == 0) {
				
				reconstruct = rbm.reconstruct(first.getFirst());
				
				if (i > 0)
					
					//for (int j = 0; j < first.numExamples(); j++) {
					for (int j = 0; j < 10; j++) {
						

						
						Matrix draw1 = first.get(j).getFirst().times(255);
						Matrix reconstructed2 = MatrixUtils.viewRowAsMatrix(reconstruct, j);
						Matrix draw2 = MatrixUtils.genBinomialDistribution(reconstructed2,1,new MersenneTwister(123)).times(255);
						
						
						DrawMnistGreyscale d = new DrawMnistGreyscale(draw1);
						d.title = "REAL";
						d.draw();
						DrawMnistGreyscale d2 = new DrawMnistGreyscale(draw2,100,100);
						d2.title = "TEST";
						d2.draw();
						Thread.sleep(5000);
						d.frame.dispose();
						d2.frame.dispose();

					}
			}
			
			rbm.train(first.getFirst(), 0.01, new Object[]{1});
			
			log.info("Training Epoch " + i + " Negative log likelihood " + rbm.getReConstructionCrossEntropy());


		}







	}

}
