package tv.floe.metronome.classification.neuralnetworks.learning;

import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.transfer.TransferFunction;

public class SigmoidDeltaLearningAlgorithm extends LeastMeanSquaresLearningAlgorithm {

	@Override
	protected void updateNetworkWeights(double[] outputError) {
		
		//System.out.println("> SigmoidDeltaLearningAlgorithm::updateNetworkWeights()");
		
		this.calculateErrorAndUpdateOutputNeurons(outputError);
		
	}

	/**
	 * This method implements weights update procedure for the output neurons
	 * Calculates delta/error and calls updateNeuronWeights to update neuron's weights
         * for each output neuron
         * 
	 * @param outputError
	 *            error vector for output neurons
	 */
	protected void calculateErrorAndUpdateOutputNeurons(double[] outputError) {
		int i = 0;
                // for all output neurons
		for(Neuron neuron : nn.getOutputNeurons()) {
                        // if error is zero, just set zero error and continue to next neuron
			if (outputError[i] == 0) {
				neuron.setError(0);
                                i++;
				continue;
			}
			
                        // otherwise calculate and set error/delta for the current neuron
			TransferFunction transferFunction = neuron.getTransferFunction();
			double neuronInput = neuron.getNetInput();
			double delta = outputError[i] * transferFunction.getDerivative(neuronInput); // delta = (d-y)*df(net)
			neuron.setError(delta);
                        
                        // and update weights of the current neuron
			this.updateNeuronWeights(neuron);				
			i++;
		} // for				
	}	
	
}
