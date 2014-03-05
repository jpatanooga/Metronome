package tv.floe.metronome.deeplearning.rbm;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.Map;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

import tv.floe.metronome.deeplearning.datasets.DataSet;
import tv.floe.metronome.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import tv.floe.metronome.deeplearning.rbm.visualization.RBMRenderer;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

public class TestRBMRenderer {

	
	private static void renderActivationsToDisk( RestrictedBoltzmannMachine rbm, String CE, int scale ) throws InterruptedException {
		
		String strCE = String.valueOf(CE).substring(0, 5);

		// Matrix hbiasMean = network.getInput().mmul(network.getW()).addRowVector(network.gethBias());
		
		//Matrix hbiasMean = MatrixUtils.addRowVector( rbm.getInput().times( rbm.connectionWeights ), rbm.getHiddenBias().viewRow(0) );
		Matrix hbiasMean = MatrixUtils.sigmoid( MatrixUtils.addRowVector( rbm.getInput().times( rbm.connectionWeights ), rbm.getHiddenBias().viewRow(0) ) );


		RBMRenderer renderer = new RBMRenderer();
		//rbm_hbias_test.renderHiddenBiases(100, 100, hbiasMean, "/tmp/Metronome/RBM/" + UUIDForRun + "/activations_" + strCE + "_ce.png");
		
		renderer.renderActivations(100, 100, hbiasMean, "/tmp/Metronome/unit_test/RBMRenderer/activations_" + strCE + "_ce.png", scale );
		
	}
	
	
	private static void renderWeightValuesToDisk( RestrictedBoltzmannMachine rbm, String CE ) throws InterruptedException {
		
		String strCE = String.valueOf(CE).substring(0, 5);

		// Matrix hbiasMean = network.getInput().mmul(network.getW()).addRowVector(network.gethBias());
		
		//Matrix hbiasMean = MatrixUtils.addRowVector( rbm.getInput().times( rbm.connectionWeights ), rbm.getHiddenBias().viewRow(0) );
		//Matrix hbiasMean = MatrixUtils.sigmoid( MatrixUtils.addRowVector( rbm.getInput().times( rbm.connectionWeights ), rbm.getHiddenBias().viewRow(0) ) );


		RBMRenderer renderer = new RBMRenderer();
		//rbm_hbias_test.renderHiddenBiases(100, 100, hbiasMean, "/tmp/Metronome/RBM/" + UUIDForRun + "/activations_" + strCE + "_ce.png");
		
		renderer.renderHistogram( rbm.connectionWeights, "/tmp/Metronome/unit_test/RBMRenderer/weight_histogram_" + strCE + "_ce.png", 10 );
		
	}	
	
	
	@Test
	public void testRBMRenders() throws InterruptedException {

		double ce = 0;
		double learningRate = 0.01;


		double[][] data_simple = new double[][]
				{
					{1,1,1,0,0,0},
					{0,0,0,1,1,1},
					{1,1,1,0,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,1,0}
					
				};
		
		Matrix input = new DenseMatrix(data_simple);		
		
		int weightScale = 1000;
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 4, null);
		rbm.useRegularization = false;
		rbm.connectionWeights = rbm.connectionWeights.times( weightScale );
		
		rbm.setInput(input);
		
		System.out.println( "Initial Weights: " );
		
		MatrixUtils.debug_print( rbm.connectionWeights );
		
		//ce = rbm.getReConstructionCrossEntropy();
		renderActivationsToDisk(rbm, "_init", weightScale );
		
			
		//rbm.trainTillConvergence(0.01, 1, input);
		rbm.trainTillConvergence( input, learningRate, new Object[]{ 1, learningRate, 10 } );

		System.out.println( "Trained Weights: " );
		
		MatrixUtils.debug_print( rbm.connectionWeights );

		ce = rbm.getReConstructionCrossEntropy();
		renderActivationsToDisk(rbm, "" + ce, weightScale );
		
		
		
		
	
	
	}
	
	@Test
	public void testMNISTRenderPath() throws InterruptedException, IOException {
		
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

		//MatrixUtils.debug_print_row( rbm.trainingDataset, 1 );

		// render base activations pre train
		
		renderActivationsToDisk(rbm, "_init", 1);		
		
		renderWeightValuesToDisk( rbm, "_init" );
		
	}
	
	@Test
	public void testComputeHistogramBucketIndex() {
		
		RBMRenderer renderer = new RBMRenderer();

		int bin = renderer.computeHistogramBucketIndex( -0.2, 0.05, -0.1, 10 );
		
		System.out.println("bin: " + bin);
		
		assertEquals( 2, bin );
		
	}
	
	@Test
	public void testGenerateHistogramBins() {
		
		double[][] data_simple = new double[][]
				{
					{1,1,1,0,0,0},
					{0,0,0,1,1,1},
					{1,1,1,0,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,1,0}
					
				};

		double[][] data_simple2 = new double[][]
				{
					{1,1,1,1,1,1},
					{0,1,1,1,1,1}
					
				};
		
		
		Matrix input = new DenseMatrix(data_simple2);		
		
		
		RBMRenderer renderer = new RBMRenderer();
/*		
		Map<Integer, Pair<String, Integer>> map = renderer.generateHistogramBuckets( input, 2 ); 
		
		for (Map.Entry<Integer, Pair<String, Integer>> entry : map.entrySet()) {
			
			Integer key = entry.getKey();
			Pair<String, Integer> value = entry.getValue();

			System.out.println(key + " => " + value.getFirst() + ",  " + value.getSecond());
			  
		}
*/		
		Map<Integer, Integer> map = renderer.generateHistogramBuckets( input, 2 );

		for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
			
			Integer key = entry.getKey();
			Integer val = entry.getValue();
			System.out.println(key + " => " + key + ",  " + val );
			  
		}
		
		Integer val_key_0 = map.get(0);
		Integer val_key_1 = map.get(1);
		
		assertEquals( 1, val_key_0.intValue() );
		assertEquals( 11, val_key_1.intValue() );
		
	}

	
	@Test
	public void testGenerateHistogramBins2() {
		
		double[][] data_simple = new double[][]
				{
					{ 0.1, 0.5, 0.25, 1.0, -0.5 },
					{ -0.4, -0.3, -0.25, -0.1, -0.5 },
					{ 0.1, 0.5, 0.5, 0.5, 0.1 },
					
				};
		
		Matrix input = new DenseMatrix(data_simple);		
		
		
		RBMRenderer renderer = new RBMRenderer();
		
		//Map<Integer, Pair<String, Integer>> map = renderer.generateHistogramBuckets( input, 2 ); 
		Map<Integer, Integer> map = renderer.generateHistogramBuckets( input, 2 );
		
//		for (Map.Entry<Integer, Pair<String, Integer>> entry : map.entrySet()) {
		for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
			
			Integer key = entry.getKey();
			Integer val = entry.getValue();
			//Pair<String, Integer> value = entry.getValue();

			//System.out.println(key + " => " + value.getFirst() + ",  " + value.getSecond());
			System.out.println(key + " => " + key + ",  " + val );
			  
		}
		
		Integer val_key_0 = map.get(0);
		Integer val_key_1 = map.get(1);
		
		
		assertEquals( 10, val_key_0.intValue() );
		assertEquals( 5, val_key_1.intValue() );
		
		
		renderer.renderHistogram(input, "/tmp/debug_render_rbm_histogram.png", 2);
		
		
	}	
	
	
}
