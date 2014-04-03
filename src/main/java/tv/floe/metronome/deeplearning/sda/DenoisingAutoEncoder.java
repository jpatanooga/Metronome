package tv.floe.metronome.deeplearning.sda;

import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseNeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.gradient.NeuralNetworkGradient;

public class DenoisingAutoEncoder extends BaseNeuralNetworkVectorized {

	@Override
	public Matrix reconstruct(Matrix x) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double lossFunction(Object[] params) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void train(Matrix input, double learningRate, Object[] params) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void trainTillConvergence(Matrix input, double lr, Object[] params) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public NeuralNetworkGradient getGradient(Object[] params) {
		// TODO Auto-generated method stub
		return null;
	}
	
	
	
	
	
	
}
