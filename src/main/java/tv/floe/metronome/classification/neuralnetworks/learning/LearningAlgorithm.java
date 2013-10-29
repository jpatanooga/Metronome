package tv.floe.metronome.classification.neuralnetworks.learning;

import java.io.Serializable;

import org.apache.mahout.math.Vector;

import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;

public abstract class LearningAlgorithm implements Serializable {

    protected NeuralNetwork nn;
    
    public LearningAlgorithm() {
    }

    public NeuralNetwork getNeuralNetwork() {
        return nn;
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        this.nn = neuralNetwork;
    }


	
    abstract public void train(Vector actual_output_values, Vector instance);

    // only used in supervised learning??
	protected void updateNetworkWeights(double[] outputError) {
		// TODO Auto-generated method stub
		
	}

	
	
	
	
}
