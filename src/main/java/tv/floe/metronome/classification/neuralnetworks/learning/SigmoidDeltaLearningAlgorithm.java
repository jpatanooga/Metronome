package tv.floe.metronome.classification.neuralnetworks.learning;

import tv.floe.metronome.classification.neuralnetworks.activation.ActivationFunction;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.activation.ActivationFunction;

public class SigmoidDeltaLearningAlgorithm extends LeastMeanSquaresLearningAlgorithm {

	@Override
	protected void updateNetworkWeights(double[] outputError) {
				
		this.calculateErrorAndUpdateOutputNeurons(outputError);
		
	}

	/**
	 * used by back propagation to update output neurons
	 * 
	 * 
	 * @param outputError
	 */
	protected void calculateErrorAndUpdateOutputNeurons(double[] outputError) {
		
		int x = 0;
		
		for (Neuron neuron : nn.getOutputNeurons()) {

			if (outputError[ x ] == 0) {
				neuron.setError( 0 );
                x++;
				continue;
			}
			
			ActivationFunction transferFunction = neuron.getActivationFunction();
			double neuronInput = neuron.getNetInput();
			double delta = outputError[ x ] * transferFunction.getDerivative( neuronInput ); // delta = (d-y)*df(net)
			neuron.setError( delta );
                        
			this.updateNeuronWeights( neuron );				
			x++; 
			
		} // for
		
	}	
	
}
