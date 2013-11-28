package tv.floe.metronome.classification.neuralnetworks.learning;

import java.util.ArrayList;

import tv.floe.metronome.classification.neuralnetworks.core.Connection;
import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.core.Weight;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.learning.adagrad.AdagradLearningRate;
import tv.floe.metronome.classification.neuralnetworks.activation.ActivationFunction;

/**
 * Basic backprop implementation
 * - adagrad and momentum learning are added in a very basic way, no elaborate class system
 * 
 * @author josh
 *
 */
public class BackPropogationLearningAlgorithm extends SigmoidDeltaLearningAlgorithm {

	boolean adagradLearningOn = false;
	boolean momentumLearningOn = false;
	
	public BackPropogationLearningAlgorithm() {
		super();
	}
	
	public void turnOnAdagradLearning() {
		this.adagradLearningOn = true;
	}
	
	public void turnOnMomentumLearning() {
		this.momentumLearningOn = true;
	}
	
    @Override
    public void setup() {
    	
    	if (this.adagradLearningOn) {
    		
    		// add stuff to the weights
    		
    		ArrayList<Layer> layers = nn.getLayers();
    		
    		
    		/*
    		for (int l = layers.size() - 2; l > 0; l--) {
    									
    			for ( Neuron neuron : layers.get( l ).getNeurons() ) {	
                                    
    				//double neuronError = this.calculateHiddenNeuronError( neuron ); 
    				
    				//neuron.setError( neuronError );
    				
    				//this.updateNeuronWeights( neuron );
    								
    			} // for
    			
    		} // for  
    		*/
    		
    		for ( int x = 1; x < this.nn.getLayersCount(); x++ ) {
	        	
                for (Neuron neuron : this.nn.getLayerByIndex(x).getNeurons()) {
                	
                    for (Connection connection : neuron.getInConnections()) {
                    	
                        connection.getWeight().trainingMetaData.put("adagrad", new AdagradLearningRate(10.0));
                        
                    }
                    
                }
    	            
    			
    		}    		
    		
    		
    		
    	}
    	
    	if (this.momentumLearningOn) {
    		
    		// add stuff to the weights
    		
    	}
    	
    	
    }
	


	/**
	 * This is the main driver method for the learning algorithms 
	 * train( ... )
	 * - calculate output of current instance
	 * - calculate error of output vs desired output
	 * 
	 */
	@Override
	protected void updateNetworkWeights(double[] outputError) {
				
		this.calculateErrorAndUpdateOutputNeurons(outputError); // via SigmoidDelta
		this.calculateErrorAndUpdateHiddenNeurons();            // implemented in this class
		
	}

	protected void calculateErrorAndUpdateHiddenNeurons() {
                
		ArrayList<Layer> layers = nn.getLayers();
		
	
		
		for (int l = layers.size() - 2; l > 0; l--) {
									
			for ( Neuron neuron : layers.get( l ).getNeurons() ) {	
                                
				double neuronError = this.calculateHiddenNeuronError( neuron ); 
				
				neuron.setError( neuronError );
				
				this.updateNeuronWeights( neuron );
								
			} // for
			
		} // for
		
	}

	protected double calculateHiddenNeuronError(Neuron neuron) {	
		
		double deltaSum = 0d;
		
		
		
		for (Connection connection : neuron.getOutConnections()) {	
			
			double delta = connection.getToNeuron().getError() * connection.getWeight().value;
			deltaSum += delta; // weighted delta sum from the next layer
		
			if (this.isMetricCollectionOn()) {
				this.metrics.incErrCalcOpCount();
			}
			
		} // for

		
		
		ActivationFunction transferFunction = neuron.getActivationFunction();
		
		
		double netInput = neuron.getNetInput(); // should we use input of this or other neuron?
		double fnDerv = transferFunction.getDerivative(netInput);
		double neuronError = fnDerv * deltaSum;
		return neuronError;
	}	
	
	
	/**
	 * Updated for adagrad
	 * 
	 * so with adagrad engaged, we are tracking/accumulating the gradient change
	 * into the AdagradLearningRate object on each weight
	 * - over time this drives the learning rate downwards
	 * 
	 */
	@Override
    protected void updateNeuronWeights(Neuron neuron) {

    	double neuronError = neuron.getError();
        double lrTemp = 0;
        AdagradLearningRate alr = null;
        
        for (Connection connection : neuron.getInConnections()) {

        	if (this.adagradLearningOn) {
        		alr = (AdagradLearningRate)connection.getWeight().trainingMetaData.get("adagrad");
        		lrTemp = alr.compute();
        	} else {
        		lrTemp = this.learningRate;
        	}
        	
        	double input = connection.getInput();
            //double weightChange = this.learningRate * neuronError * input;
        	double weightChange = lrTemp * neuronError * input;

            Weight weight = connection.getWeight();

            if (this.isInBatchMode() == false) {             
                weight.weightChange = weightChange;                
                weight.value += weightChange;
            } else { 
                weight.weightChange += weightChange;
            }
            
        	if (this.adagradLearningOn) {
        		alr = (AdagradLearningRate)connection.getWeight().trainingMetaData.get("adagrad");
        		alr.addLastIterationGradient(weightChange);
        	}            
            
            if (this.isMetricCollectionOn()) {
            	this.metrics.incWeightOpCount();
            }
        }
    }	
	
	
	
}
