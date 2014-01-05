package tv.floe.metronome.deeplearning.neuralnetwork.core;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;

public class BaseMultiLayerNeuralNetworkVectorized {

	public int inputNeuronCount;
	
	//the hidden layer sizes at each layer
	public int[] hiddenLayerSizes;
	
	
	public int outputNeuronCount;
	public int numberLayers;
	
	//the hidden layers
	public LayerVectorized[] hiddenLayers;	
	
	public RandomGenerator randomGenerator;
	
	// the input data ---- how is this going to be handled?
	// how was it handled with the OOP-MLPN version?
	Matrix input = null;
	
	public double learningRateUpdate = 0.95;
	
	/**
	 * CTOR
	 * 
	 */
	public BaseMultiLayerNeuralNetworkVectorized() {
		
	}
	
	
	public void initLayers() {
		
		
	}
	
	
}
