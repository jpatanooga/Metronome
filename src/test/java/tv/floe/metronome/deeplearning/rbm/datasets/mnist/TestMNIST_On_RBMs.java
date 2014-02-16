package tv.floe.metronome.deeplearning.rbm.datasets.mnist;

import static org.junit.Assert.*;

import java.util.UUID;

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

	private String UUIDForRun = UUID.randomUUID().toString();
	
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

		Thread.sleep(2000);
		d.frame.dispose();
		d2.frame.dispose();
		
		
		
	}
	

	private void renderExampleToDisk( Matrix draw1, Matrix reconstructed2, Matrix draw2, String number, String CE ) throws InterruptedException {

		DrawMnistGreyscale d = new DrawMnistGreyscale(draw1);
//		d.title = "REAL";
		d.saveToDisk("/tmp/Metronome/RBM/" + UUIDForRun + "/" + CE + "_ce_" + number + "_real.png");
		
		DrawMnistGreyscale d2 = new DrawMnistGreyscale( draw2, 100, 100 );
//		d2.title = "TEST";
		d2.saveToDisk("/tmp/Metronome/RBM/" + UUIDForRun + "/" + CE + "_ce_" + number + "_test.png");

		
		
/*		Thread.sleep(2000);
		d.frame.dispose();
		d2.frame.dispose();
	*/	
		
		
	}	
	
	public void renderBatchOfReconstructions(RestrictedBoltzmannMachine rbm, DataSet input, boolean toDisk, String CE) throws InterruptedException {
		

		Matrix reconstruct_all = rbm.reconstruct( input.getFirst() );

//		log.info("Negative log likelihood " + rbm.getReConstructionCrossEntropy());

		System.out.println(" ----- Visualizing Reconstructions ------");
		
		for (int j = 0; j < 10; j++) {
			
			// get the actual image we're looking at
			Matrix draw1 = input.get(j).getFirst().times(255);
			
			// get the reconstruction row that matches this image
			Matrix reconstructed_row_image = MatrixUtils.viewRowAsMatrix(reconstruct_all, j);
			
			// now generate a new image based on the reconstruction probabilities
			Matrix draw2 = MatrixUtils.genBinomialDistribution( reconstructed_row_image, 1, new MersenneTwister(123) ).times(255);
		
			if (toDisk) {
				
				renderExampleToDisk(draw1, reconstructed_row_image, draw2, String.valueOf(j), CE);
				
			} else {
				renderExample(draw1, reconstructed_row_image, draw2);
			}
			
		}
		
	}
	
	@Test
	public void testMnist() throws Exception {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(100,100);
		MersenneTwister rand = new MersenneTwister(123);

		double learningRate = 0.005;
		
		int[] batchSteps = { 250, 200, 150, 100, 50, 25, 5 };
		
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
		
		//for(int i = 0; i < 2; i++) {
		int epoch = 0;
		
		
		for (int stepIndex = 0; stepIndex < batchSteps.length; stepIndex++ ) {
		
			int minCrossEntropy = batchSteps[ stepIndex ];
			
			while ( rbm.getReConstructionCrossEntropy() > minCrossEntropy) {
				
				System.out.println("Epoch " + epoch + " Negative Log Likelhood: " + rbm.getReConstructionCrossEntropy() );
				
				//rbm.trainTillConvergence( first.getFirst(), learningRate, new Object[]{ 1 } );
				rbm.trainTillConvergence(learningRate, 1, first.getFirst());
				
				epoch++;
				
			}

			System.out.println(" ----- Visualizing Reconstructions Step " + minCrossEntropy + " CE ------");
			
			renderBatchOfReconstructions( rbm, first, true, String.valueOf(rbm.getReConstructionCrossEntropy()) );
			
			
		}
		
		/*
		while ( rbm.getReConstructionCrossEntropy() > 250) {
			
			System.out.println("Epoch " + epoch + " Negative Log Likelhood: " + rbm.getReConstructionCrossEntropy() );
			
			//rbm.trainTillConvergence( first.getFirst(), learningRate, new Object[]{ 1 } );
			rbm.trainTillConvergence(learningRate, 1, first.getFirst());
			
			epoch++;
			
		}

		System.out.println(" ----- Visualizing Reconstructions sub 250 CE ------");
		
		renderBatchOfReconstructions( rbm, first, true, String.valueOf(rbm.getReConstructionCrossEntropy()) );
		
		while ( rbm.getReConstructionCrossEntropy() > 200) {
			
			System.out.println("Epoch " + epoch + " Negative Log Likelhood: " + rbm.getReConstructionCrossEntropy() );
			
			//rbm.trainTillConvergence( first.getFirst(), learningRate, new Object[]{ 1 } );
			rbm.trainTillConvergence(learningRate, 1, first.getFirst());
			
			epoch++;
			
		}		
		
		System.out.println(" ----- Visualizing Reconstructions sub 200 CE ------");
		
//		renderBatchOfReconstructions( rbm, first );
		renderBatchOfReconstructions( rbm, first, true, String.valueOf(rbm.getReConstructionCrossEntropy()) );
		

		
		while ( rbm.getReConstructionCrossEntropy() > 50) {
			
			System.out.println("Epoch " + epoch + " Negative Log Likelhood: " + rbm.getReConstructionCrossEntropy() );
			
			//rbm.trainTillConvergence( first.getFirst(), learningRate, new Object[]{ 1 } );
			rbm.trainTillConvergence(learningRate, 1, first.getFirst());
			
			epoch++;
			
		}		
		
		System.out.println(" ----- Visualizing Reconstructions sub 50 CE ------");
		
		renderBatchOfReconstructions( rbm, first, true, String.valueOf(rbm.getReConstructionCrossEntropy()) );
		

*/



	}

}
