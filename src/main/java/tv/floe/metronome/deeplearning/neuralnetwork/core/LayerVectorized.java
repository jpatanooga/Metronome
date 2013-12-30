package tv.floe.metronome.deeplearning.neuralnetwork.core;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;

public class LayerVectorized {
	
	public Matrix weights;
	public Matrix biasTerms;
	public RandomGenerator rndNumGenerator;
	public Matrix input;
	
	public LayerVectorized() {
		
		
	}
	
	/**
	 * In the vectorized implementation we want to hook the previous layer's
	 * work in progress into this layer
	 * 
	 * In the case of the first layer, its just the matrix of input samples
	 * 
	 * @param layerInput
	 */
	public void setInput(Matrix layerInput) {
		
	}
	
	
	public Matrix getVectorizedOutputMatrix() {
		
		return null;
	}
	
	

}
