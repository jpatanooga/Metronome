package tv.floe.metronome.classification.neuralnetworks.learning;

import org.apache.mahout.math.Vector;

import tv.floe.metronome.classification.neuralnetworks.core.Connection;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.Weight;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class LeastMeanSquaresLearningAlgorithm extends LearningAlgorithm {

	protected double learningRate = 0.1d;
	
    protected transient double totalNetworkError;
    protected transient double totalSquaredErrorSum;
    protected transient double previousEpochError;
    protected double maxError = 0.01d;
    private double minErrorChange = Double.POSITIVE_INFINITY;
    private int minErrorChangeIterationsLimit = Integer.MAX_VALUE;
    private transient int minErrorChangeIterationsCount;
    private boolean batchMode = false;
        
    public double getTotalSquaredError() {
    	return this.totalSquaredErrorSum;
    }
    
    public void clearTotalSquaredError() {
    	
    	this.totalSquaredErrorSum = 0;
    	
    }
	
	public void train(Vector actual_output_vector, Vector training_instance) {
		
        try {
			this.nn.setInputVector( training_instance );
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
                
        this.nn.calculate();
        
        Vector output_vec = this.nn.getOutputVector();
        
        double[] outputError = this.calculateOutputError( actual_output_vector, output_vec );

        this.addToSquaredErrorSum(outputError);

        this.updateNetworkWeights(outputError);
        
		
	}
	

    @Override
    protected void updateNetworkWeights(double[] outputError) {
        int i = 0;
        for (Neuron neuron : nn.getOutputNeurons()) {
            neuron.setError(outputError[i]); // set the neuron error, as difference between desired and actual output 
            this.updateNeuronWeights(neuron); // and update neuron weights
            i++;
        }
    }

    protected void updateNeuronWeights(Neuron neuron) {

    	double neuronError = neuron.getError();
        
        for (Connection connection : neuron.getInConnections()) {

        	double input = connection.getInput();
            double weightChange = this.learningRate * neuronError * input;

            Weight weight = connection.getWeight();

            if (this.isInBatchMode() == false) {             
                weight.weightChange = weightChange;                
                weight.value += weightChange;
            } else { 
                weight.weightChange += weightChange;
            }
        }
    }	
    
    protected boolean hasReachedStopCondition() {
        return (this.totalNetworkError < this.maxError) || this.errorChangeStalled();
    }

    protected boolean errorChangeStalled() {
        double absErrorChange = Math.abs(previousEpochError - totalNetworkError);

        if (absErrorChange <= this.minErrorChange) {
            this.minErrorChangeIterationsCount++;

            if (this.minErrorChangeIterationsCount >= this.minErrorChangeIterationsLimit) {
                return true;
            }
        } else {
            this.minErrorChangeIterationsCount = 0;
        }

        return false;
    }

    protected double[] calculateOutputError(Vector desiredOutputVec, Vector currentOutputVec) {
        
    	double[] outputError = new double[ desiredOutputVec.size() ];
        
        for (int i = 0; i < currentOutputVec.size(); i++) {

        	outputError[ i ] = desiredOutputVec.get(i) - currentOutputVec.get(i);
        	
        }
        
        return outputError;
    }

    protected void addToSquaredErrorSum(double[] outputError) {
        
    	double outputErrorSqrSum = 0.0d;
    	
        for (double error : outputError) {
        
        	outputErrorSqrSum += (error * error) * 0.5d; // a;so multiply with 1/trainingSetSize  1/2n * (...)
        
        	//System.out.println("> err^2 " + ((error * error)) );
        	
        }

        this.totalSquaredErrorSum += outputErrorSqrSum;
    }
    
    public boolean isInBatchMode() {
        return batchMode;
    }

    public void setBatchMode(boolean batchMode) {
        this.batchMode = batchMode;
    }    
    
    
    
}
