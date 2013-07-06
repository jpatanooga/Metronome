package tv.floe.metronome.classification.neuralnetworks.learning;

import java.util.ArrayList;

import tv.floe.metronome.classification.neuralnetworks.core.Connection;
import tv.floe.metronome.classification.neuralnetworks.core.Layer;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;
import tv.floe.metronome.classification.neuralnetworks.transfer.TransferFunction;

public class BackPropogationLearningAlgorithm extends SigmoidDeltaLearningAlgorithm {

	public BackPropogationLearningAlgorithm() {
		super();
	}


	@Override
	protected void updateNetworkWeights(double[] outputError) {
		
		//System.out.println( "> back prop > update network weights" );
		
		this.calculateErrorAndUpdateOutputNeurons(outputError); // inherited from SigmoidDeltaRule
		this.calculateErrorAndUpdateHiddenNeurons();            // implemented in this class
	}

	protected void calculateErrorAndUpdateHiddenNeurons() {
                
		ArrayList<Layer> layers = nn.getLayers();
		
		for (int l = layers.size() - 2; l > 0; l--) {
			
			int neuron_id = 0;
			
			//System.out.println( "> Layer " + l );
			
			for ( Neuron neuron : layers.get( l ).getNeurons() ) {	
                                
				// calculate the neuron's error (delta)
				double neuronError = this.calculateHiddenNeuronError( neuron ); 
				
			//	System.out.println( "> Layer " + l + ", n: " + neuron_id + " Err: " + neuronError  );
				
				neuron.setError( neuronError );
				
				this.updateNeuronWeights( neuron );
				
				neuron_id++;
				
			} // for
			
		} // for
		
	}

	protected double calculateHiddenNeuronError(Neuron neuron) {	
		
		double deltaSum = 0d;
		
		for (Connection connection : neuron.getOutConnections()) {	
			
			double delta = connection.getToNeuron().getError() * connection.getWeight().value;
			deltaSum += delta; // weighted delta sum from the next layer
		
		} // for

		TransferFunction transferFunction = neuron.getTransferFunction();
		
	//	System.out.println(">TF: " + transferFunction.getClass());
	//	System.out.println(">class: " + neuron.getClass() );
		
		double netInput = neuron.getNetInput(); // should we use input of this or other neuron?
		double f1 = transferFunction.getDerivative(netInput);
		double neuronError = f1 * deltaSum;
		return neuronError;
	}	
	
}
