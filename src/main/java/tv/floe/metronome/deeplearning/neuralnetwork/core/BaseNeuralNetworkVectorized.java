package tv.floe.metronome.deeplearning.neuralnetwork.core;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;

/**
 * Based on the NN design by Adam Gibson
 * 
 * @author josh
 *
 */
//public abstract class BaseNeuralNetwork implements NeuralNetwork,Persistable {
public abstract class BaseNeuralNetworkVectorized implements NeuralNetworkVectorized {

	//public int inputNeuronCount;
	
	public int numberVisibleNeurons;
	public int numberHiddenNeurons;
	
	public Matrix connectionWeights;
	
	public Matrix hiddenBiasNeurons;
	public Matrix visibleBiasNeurons;

	
	//the hidden layer sizes at each layer
	public int[] hiddenLayerSizes;
	
	
	//public int outputNeuronCount;
	public int numberLayers;
	
	//the hidden layers
	public LayerVectorized[] hiddenLayers;	
	
	public RandomGenerator randomGenerator;
	
	// the input data ---- how is this going to be handled?
	// how was it handled with the OOP-MLPN version?
	Matrix input = null;	

	public double sparsity = 0.01;
	/* momentum for learning */
	public double momentum = 0.1;
	/* L2 Regularization constant */
	public double l2 = 0.1;
	
	
	@Override
	public int getnVisible() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setnVisible(int nVisible) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int getnHidden() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setnHidden(int nHidden) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Matrix getConnectionWeights() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setConnectionWeights(Matrix weights) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Matrix getHiddenBias() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void sethBias(Matrix hiddenBias) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Matrix getVisibleBias() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setVisibleBias(Matrix visibleBias) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public RandomGenerator getRng() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setRng(RandomGenerator rng) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Matrix getInput() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setInput(Matrix input) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double getSparsity() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setSparsity(double sparsity) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double getL2() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setL2(double l2) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double getMomentum() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setMomentum(double momentum) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void trainTillConvergence(Matrix input, double lr, Object[] params) {
		// TODO Auto-generated method stub
		
	}	
}
