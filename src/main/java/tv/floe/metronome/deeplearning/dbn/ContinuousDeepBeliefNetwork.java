package tv.floe.metronome.deeplearning.dbn;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.rbm.ContinuousRestrictedBoltzmannMachine;
import tv.floe.metronome.deeplearning.neuralnetwork.core.NeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.rbm.RestrictedBoltzmannMachine;


public class ContinuousDeepBeliefNetwork extends DeepBeliefNetwork {

	@Override
	public NeuralNetworkVectorized createPreTrainingLayer(Matrix input,
			int nVisible, int nHidden, Matrix weights, Matrix hBias,
			Matrix vBias, RandomGenerator rng, int index) {
	
		if (index == 0) {
			return new ContinuousRestrictedBoltzmannMachine(input, nVisible, nHidden, weights, hBias, vBias, rng);
		} else {
			return new RestrictedBoltzmannMachine(input, nVisible, nHidden, weights, hBias, vBias, rng);
		}
	}

	@Override
	public NeuralNetworkVectorized[] createNetworkLayers(int numLayers) {
	
		return new RestrictedBoltzmannMachine[numLayers];	
		
	}
	
	
}
