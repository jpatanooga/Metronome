package tv.floe.metronome.deeplearning.sda;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseMultiLayerNeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.core.NeuralNetworkVectorized;

public class StackedDenoisingAutoEncoder extends BaseMultiLayerNeuralNetworkVectorized {

	@Override
	public NeuralNetworkVectorized createPreTrainingLayer(Matrix input,
			int nVisible, int nHidden, Matrix weights, Matrix hbias,
			Matrix vBias, RandomGenerator rng, int index) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void trainNetwork(Matrix input, Matrix labels, Object[] otherParams) {
		// TODO Auto-generated method stub
		
	}
	

}
