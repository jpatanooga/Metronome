package tv.floe.metronome.deeplearning.rbm;

import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseNeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.optimize.NeuralNetworkOptimizer;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;


public class RestrictedBoltzmannMachineOptimizer extends NeuralNetworkOptimizer {

	
	private static final long serialVersionUID = 3676032651650426749L;
	protected int k = -1;
	protected int numTimesIterated = 0;
	
	public RestrictedBoltzmannMachineOptimizer(BaseNeuralNetworkVectorized network, double lr,
			Object[] trainingParams) {
		super(network, lr, trainingParams);
	}

	/**
	 * TODO: double check  this against the stock RBM impl
	 * 
	 * 
	 */
	@Override
	public void getValueGradient(double[] buffer) {
		int k = (Integer) extraParams[0];
		numTimesIterated++;
		
		//System.out.println("k: " + k);
		
		//adaptive k based on the number of iterations.
		//typically over time, you want to increase k.
		if(this.k <= 0)
			this.k = k;
		if(numTimesIterated % 10 == 0) {
			this.k++;
		}
		
		
		//Don't go over 15
		if(this.k >= 15) 
		     this.k = 15;
		
		k = this.k;
		/*
		 * Cost and updates dictionary.
		 * This is the update rules for weights and biases
		 */
		RestrictedBoltzmannMachine r = (RestrictedBoltzmannMachine) network;
		Pair<Matrix,Matrix> probHidden = r.sampleHiddenGivenVisible(r.trainingDataset);

		/*
		 * Start the gibbs sampling.
		 */
		Matrix chainStart = probHidden.getSecond();

		/*
		 * Note that at a later date, we can explore alternative methods of 
		 * storing the chain transitions for different kinds of sampling
		 * and exploring the search space.
		 */
		Pair<Pair<Matrix,Matrix>,Pair<Matrix,Matrix>> matrices = null;
		//negative visible means or expected values
		@SuppressWarnings("unused")
		Matrix nvMeans = null;
		//negative value samples
		Matrix nvSamples = null;
		//negative hidden means or expected values
		Matrix nhMeans = null;
		//negative hidden samples
		Matrix nhSamples = null;

		/*
		 * K steps of gibbs sampling. THis is the positive phase of contrastive divergence.
		 * 
		 * There are 4 matrices being computed for each gibbs sampling.
		 * The samples from both the positive and negative phases and their expected values or averages.
		 * 
		 */

		for (int i = 0; i < k; i++) {


			if (i == 0) { 
				//matrices = r.gibbhVh(chainStart);
				matrices = r.gibbsSamplingStepFromHidden( chainStart );
			} else {
				//matrices = r.gibbhVh(nhSamples);
				matrices = r.gibbsSamplingStepFromHidden( nhSamples );
			}
			
			//get the cost updates for sampling in the chain after k iterations
			nvMeans = matrices.getFirst().getFirst();
			nvSamples = matrices.getFirst().getSecond();
			nhMeans = matrices.getSecond().getFirst();
			nhSamples = matrices.getSecond().getSecond();
		}
		
		// TODO: check this against the stock CD implementation

		/*
		 * Update gradient parameters
		 */
		//Matrix wAdd = r.input.transpose().mmul(probHidden.getSecond()).sub(nvSamples.transpose().mmul(nhMeans)).mul(lr).mul(0.1);
		Matrix wAdd = r.trainingDataset.transpose().times(probHidden.getSecond()).minus(nvSamples.transpose().times(nhMeans)).times(lr).times(0.1);

		//Matrix  vBiasAdd = mean(r.trainingDataset.minus(nvSamples), 0).mul(lr);
		Matrix  vBiasAdd = MatrixUtils.mean(r.trainingDataset.minus(nvSamples), 0).times(lr);


		//update rule: the expected values of the hidden input - the negative hidden  means adjusted by the learning rate
		Matrix hBiasAdd = MatrixUtils.mean(probHidden.getSecond().minus(nhMeans), 0).times(lr);
		
		int idx = 0;
		
		
		for (int i = 0; i < MatrixUtils.length(wAdd); i++) { 
		
			buffer[ idx++ ] = MatrixUtils.getElement(wAdd, i);
			
		}
		
		
		for (int i = 0; i < MatrixUtils.length( vBiasAdd ); i++) {
			
			buffer[ idx++ ] = MatrixUtils.getElement( vBiasAdd, i );
			
		}
		
		for (int i = 0; i < MatrixUtils.length( hBiasAdd ); i++) {
		
			buffer[ idx++ ] = MatrixUtils.getElement( hBiasAdd, i );
			
		}
		
		
		int wAddLen = MatrixUtils.length(wAdd);
		int vBiasLen = MatrixUtils.length(vBiasAdd);
		int hBiasLen = MatrixUtils.length(hBiasAdd);
		
		//System.out.println("> Total buff len: " + (wAddLen + vBiasLen + hBiasLen ) );
	
	}
	
}