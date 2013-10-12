package tv.floe.metronome.classification.neuralnetworks.learning;

import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.transfer.TransferFunction;

public class SigmoidDeltaLearningAlgorithm extends LeastMeanSquaresLearningAlgorithm {

	@Override
	protected void updateNetworkWeights(double[] outputError) {
				
		this.calculateErrorAndUpdateOutputNeurons(outputError);
		
	}

	/**
	 * used by back propogation to update output neurons
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
			
			TransferFunction transferFunction = neuron.getTransferFunction();
			double neuronInput = neuron.getNetInput();
			double delta = outputError[ x ] * transferFunction.getDerivative( neuronInput ); // delta = (d-y)*df(net)
			neuron.setError( delta );
                        
			this.updateNeuronWeights( neuron );				
			x++; 
			
		} // for
		
	}	
	
}
