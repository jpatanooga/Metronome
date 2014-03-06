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
import tv.floe.metronome.deeplearning.rbm.visualization.RBMRenderer;
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
	

	private void renderExampleToDisk( Matrix draw1, Matrix reconstructed2, Matrix draw2, String number, String CE, boolean renderRealImage ) throws InterruptedException {

		String strCE = String.valueOf(CE).substring(0, 5);
		
		DrawMnistGreyscale d = new DrawMnistGreyscale(draw1);
//		d.title = "REAL";
		if (renderRealImage) {
			d.saveToDisk("/tmp/Metronome/RBM/" + UUIDForRun + "/" + number + "/" + number + "_real.png");
		}
		
		DrawMnistGreyscale d2 = new DrawMnistGreyscale( draw2, 100, 100 );
//		d2.title = "TEST";
		d2.saveToDisk("/tmp/Metronome/RBM/" + UUIDForRun + "/" + number + "/" + strCE + "_ce_" + number + "_test.png");

		//RBMRenderer rbm_hbias_test = new RBMRenderer();
		//rbm_hbias_test.renderHiddenBiases(100, 100, draw2, "/tmp/Metronome/RBM/" + UUIDForRun + "/" + number + "/RBM_RENDER_TEST_" + strCE + "_ce_" + number + "_test.png");
		
		
/*		Thread.sleep(2000);
		d.frame.dispose();
		d2.frame.dispose();
	*/	
		
		
	}	
	

	private void renderhBiasToDisk( RestrictedBoltzmannMachine rbm, String CE ) throws InterruptedException {

		String strCE = String.valueOf(CE).substring(0, 5);
		
		

		RBMRenderer rbm_hbias_test = new RBMRenderer();
		rbm_hbias_test.renderHiddenBiases(100, 100, rbm.hiddenBiasNeurons, "/tmp/Metronome/RBM/" + UUIDForRun + "/hbias_" + strCE + "_ce.png");
		
		
	}		
	
	private void renderActivationsToDisk( RestrictedBoltzmannMachine rbm, String CE ) throws InterruptedException {
		
		String strCE = CE;
		if (CE.equals("init") == false) {
			strCE = String.valueOf(CE).substring(0, 5);
		}

		// Matrix hbiasMean = network.getInput().mmul(network.getW()).addRowVector(network.gethBias());
		
		Matrix hbiasMean = MatrixUtils.sigmoid( MatrixUtils.addRowVector( rbm.getInput().times( rbm.connectionWeights ), rbm.getHiddenBias().viewRow(0) ) );

		RBMRenderer renderer = new RBMRenderer();
		//rbm_hbias_test.renderHiddenBiases(100, 100, hbiasMean, "/tmp/Metronome/RBM/" + UUIDForRun + "/activations_" + strCE + "_ce.png");
		
		renderer.renderActivations(100, 100, hbiasMean, "/tmp/Metronome/RBM/" + UUIDForRun + "/activations_" + strCE + "_ce.png", 1);
		
	}
	
	
	private void renderWeightValuesToDisk( RestrictedBoltzmannMachine rbm, String CE ) throws InterruptedException {
		
		//String strCE = String.valueOf(CE).substring(0, 5);
		String strCE = CE;
		if (CE.equals("init") == false) {
			strCE = String.valueOf(CE).substring(0, 5);
		}
		

		// Matrix hbiasMean = network.getInput().mmul(network.getW()).addRowVector(network.gethBias());
		
		//Matrix hbiasMean = MatrixUtils.addRowVector( rbm.getInput().times( rbm.connectionWeights ), rbm.getHiddenBias().viewRow(0) );
		//Matrix hbiasMean = MatrixUtils.sigmoid( MatrixUtils.addRowVector( rbm.getInput().times( rbm.connectionWeights ), rbm.getHiddenBias().viewRow(0) ) );


		RBMRenderer renderer = new RBMRenderer();
		//rbm_hbias_test.renderHiddenBiases(100, 100, hbiasMean, "/tmp/Metronome/RBM/" + UUIDForRun + "/activations_" + strCE + "_ce.png");
		
		// "/tmp/Metronome/RBM/" + UUIDForRun + "/activations_" + strCE + "_ce.png"
		renderer.renderHistogram( rbm.connectionWeights, "/tmp/Metronome/RBM/" + UUIDForRun + "/weight_histogram_" + strCE + "_ce.png", 10 );
		
	}	
	
	private void renderFiltersToDisk( RestrictedBoltzmannMachine rbm, String CE ) throws Exception {
		
		//String strCE = String.valueOf(CE).substring(0, 5);
		String strCE = CE;
		if (CE.equals("init") == false) {
			strCE = String.valueOf(CE).substring(0, 5);
		}
		

		RBMRenderer renderer = new RBMRenderer();
		
		//renderer.renderHistogram( rbm.connectionWeights, "/tmp/Metronome/unit_test/RBMRenderer/weight_histogram_" + strCE + "_ce.png", 10 );
		renderer.renderFilters(rbm.connectionWeights, "/tmp/Metronome/RBM/" + UUIDForRun + "/filters_" + strCE + "_ce.png", 28, 28 );
		
	}		
	
	public void renderBatchOfReconstructions(RestrictedBoltzmannMachine rbm, DataSet input, boolean toDisk, String CE, boolean renderRealImage) throws Exception {
		

		Matrix reconstruct_all = rbm.reconstruct( input.getFirst() );

		log.info("Negative log likelihood " + rbm.getReConstructionCrossEntropy());

		System.out.println(" ----- Visualizing Reconstructions ------");
		
		for (int j = 0; j < 10; j++) {
			
			// get the actual image we're looking at
			Matrix draw1 = input.get(j).getFirst().times(255);
			
			// get the reconstruction row that matches this image
			Matrix reconstructed_row_image = MatrixUtils.viewRowAsMatrix(reconstruct_all, j);
			
			// now generate a new image based on the reconstruction probabilities
			Matrix draw2 = MatrixUtils.genBinomialDistribution( reconstructed_row_image, 1, new MersenneTwister(123) ).times(255);
		
			if (toDisk) {
				
//				System.out.println("Label: " + input.get(j).getSecond().viewRow(0).maxValueIndex() );
	//			MatrixUtils.debug_print( input.get(j).getSecond() );
				
				renderExampleToDisk(draw1, reconstructed_row_image, draw2, String.valueOf( input.get(j).getSecond().viewRow(0).maxValueIndex() ), CE, renderRealImage);
				
				 
				
			
				
			} else {
				renderExample(draw1, reconstructed_row_image, draw2);
			}
			
		}

//		renderhBiasToDisk( rbm.hiddenBiasNeurons, String.valueOf( input.get(j).getSecond().viewRow(0).maxValueIndex() ), CE, renderRealImage);
		
		//this.renderhBiasToDisk(rbm, CE);
		this.renderActivationsToDisk(rbm, CE);
		this.renderWeightValuesToDisk(rbm, CE);
		this.renderFiltersToDisk(rbm, CE);
		
	}
	
	@Test
	public void testMnist() throws Exception {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(100,200);
		MersenneTwister rand = new MersenneTwister(123);

		double learningRate = 0.001;
		
		int[] batchSteps = { 250, 200, 150, 100, 50, 25, 5 };
		
		DataSet first = fetcher.next();
/*
		RestrictedBoltzmannMachine da = new RBM.Builder().numberOfVisible(784).numHidden(400).withRandom(rand).renderWeights(1000)
				.useRegularization(false)
				.withMomentum(0).build();
*/
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine( 784, 400, null );
		rbm.useRegularization = false;
		//rbm.scaleWeights( 1000 );
		rbm.momentum = 0 ;
		rbm.sparsity = 0.01;
		// TODO: investigate "render weights"



		rbm.trainingDataset = first.getFirst();

		//MatrixUtils.debug_print( rbm.trainingDataset );

		// render base activations pre train
		
		this.renderActivationsToDisk(rbm, "init");
		this.renderWeightValuesToDisk(rbm, "init");
		this.renderFiltersToDisk(rbm, "init");
		
		
		System.out.println(" ----- Training ------");
		
		//for(int i = 0; i < 2; i++) {
		int epoch = 0;
		
		System.out.println("Epoch " + epoch + " Negative Log Likelhood: " + rbm.getReConstructionCrossEntropy() );
		
		for (int stepIndex = 0; stepIndex < batchSteps.length; stepIndex++ ) {
		
			int minCrossEntropy = batchSteps[ stepIndex ];
			
			while ( rbm.getReConstructionCrossEntropy() > minCrossEntropy) {
				
				System.out.println("Epoch " + epoch + " Negative Log Likelhood: " + rbm.getReConstructionCrossEntropy() );
				
				//rbm.trainTillConvergence( first.getFirst(), learningRate, new Object[]{ 1 } );
				//rbm.trainTillConvergence(learningRate, 1, first.getFirst());
				// new Object[]{1,0.01,1000}
				rbm.trainTillConvergence(first.getFirst(), learningRate, new Object[]{ 1, learningRate, 10 } );
				
				epoch++;
				
			}

			System.out.println(" ----- Visualizing Reconstructions Step " + minCrossEntropy + " CE ------");
			
			if ( stepIndex == 0 ) {
				renderBatchOfReconstructions( rbm, first, true, String.valueOf(rbm.getReConstructionCrossEntropy()), true );
			} else {
				renderBatchOfReconstructions( rbm, first, true, String.valueOf(rbm.getReConstructionCrossEntropy()), false );
			}
			
			
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
