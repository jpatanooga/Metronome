package tv.floe.metronome.classification.neuralnetworks.learning;

import java.io.Serializable;

import org.apache.mahout.math.Vector;

import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.learning.metrics.LearningAlgorithmMetrics;

public abstract class LearningAlgorithm implements Serializable {

    protected NeuralNetwork nn;
    
    private boolean metricsOn = false;
    protected LearningAlgorithmMetrics metrics = new LearningAlgorithmMetrics();
    
    public LearningAlgorithm() {
    }

    public NeuralNetwork getNeuralNetwork() {
        return nn;
    }
    
    public LearningAlgorithmMetrics getMetrics() {
    	return this.metrics;
    }
    
    public void turnMetricsOn() {
    	this.metricsOn = true;
    }

    public void turnMetricsOff() {
    	this.metricsOn = false;
    }
    
    public boolean isMetricCollectionOn() {
    	return this.metricsOn;
    }

    
    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        this.nn = neuralNetwork;
    }


	abstract public void setup();
	
    abstract public void train(Vector actual_output_values, Vector instance);

    // only used in supervised learning??
	protected void updateNetworkWeights(double[] outputError) {
		// TODO Auto-generated method stub
		
	}

	
	
	
	
}
