package tv.floe.metronome.deeplearning.sda;

import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseNeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.gradient.NeuralNetworkGradient;
import tv.floe.metronome.types.Pair;

public class DenoisingAutoEncoder extends BaseNeuralNetworkVectorized {
    /**
     * Sample hidden mean and sample
     * given visible
     *
     * @param v the  the visible input
     * @return a pair with mean, sample
     */
    @Override
    public Pair<Matrix, Matrix> sampleHiddenGivenVisible(Matrix v) {
        return null;
    }

    /**
     * Sample visible mean and sample
     * given hidden
     *
     * @param h the  the hidden input
     * @return a pair with mean, sample
     */
    @Override
    public Pair<Matrix, Matrix> sampleVisibleGivenHidden(Matrix h) {
        return null;
    }

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
