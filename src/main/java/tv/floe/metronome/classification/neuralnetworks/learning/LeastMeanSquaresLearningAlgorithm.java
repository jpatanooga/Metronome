package tv.floe.metronome.classification.neuralnetworks.learning;

import org.apache.mahout.math.Vector;

import tv.floe.metronome.classification.neuralnetworks.core.Connection;
import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.classification.neuralnetworks.core.Weight;
import tv.floe.metronome.classification.neuralnetworks.core.neurons.Neuron;

public class LeastMeanSquaresLearningAlgorithm extends LearningAlgorithm {

	protected double learningRate = 0.1d;
	
	// per epoch
    protected transient double totalNetworkMeanSquaredError;
    
    protected transient double totalSquaredErrorSum;
    protected transient double prevEpochTotalError = 0;
    // ---
    //protected double maxError = 0.01d;
    private double stallMinErrorDelta = 0.000001;
    
    private int maxConsecutivelyStalledEpochs = 200; //Integer.MAX_VALUE;
    private int consecutivelyStalledEpochCounter = 0;
    
    private boolean batchMode = false;
    private long recordsSeenDuringEpock = 0;
    private double trainingErrorThreshold = 0.02d;
        
    
    public void setStallDetectionParams(double minErrorChange, int maxStalledEpochs) {
    	this.stallMinErrorDelta = minErrorChange;
    	this.maxConsecutivelyStalledEpochs = maxStalledEpochs;
    }
    
    public int getSetMaxStalledEpochs() {
    	return this.maxConsecutivelyStalledEpochs;
    }
    
    
    public double getTotalSquaredError() {
    	return this.totalSquaredErrorSum;
    }
    
    public void clearTotalSquaredError() {
    	
    	this.totalSquaredErrorSum = 0;
    	this.recordsSeenDuringEpock = 0;
    	
    }
    
    // needed for certain training types
    @Override
    public void setup() {
    	
    }
	
    @Override
	public void train(Vector actual_output_vector, Vector training_instance) {
		
        try {
			this.nn.setInputVector( training_instance );
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        this.recordsSeenDuringEpock++;
                
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
            
            if (this.isMetricCollectionOn()) {
            	this.metrics.incWeightOpCount();
            }
        }
    }	
    
    protected boolean hasReachedStopCondition() {
        //return (this.totalNetworkMeanSquaredError < this.maxError) || this.checkForLearningStallOut();
    	return (this.totalNetworkMeanSquaredError < this.trainingErrorThreshold) || this.checkForLearningStallOut();
    }

    public boolean hasHitMinErrorThreshold() {
    	
    	if (this.recordsSeenDuringEpock < 1) {
    		return false;
    	}
    	//System.out.println("> Debug-RMSE: " + this.calcRMSError() + " < " + this.trainingErrorThreshold);
    	double rmse = this.calcRMSError();
    	if (Double.isNaN(rmse) ) {
    		return false;
    	}
        return (rmse < this.trainingErrorThreshold);
    }
    
    /**
     * Check to see if learning has stalled out
     * - called 1x per epoch (at end)
     * 
     * 
     * @return
     */
    public boolean checkForLearningStallOut() {
    	
        double errorDelta = Math.abs(this.prevEpochTotalError - this.totalNetworkMeanSquaredError);

        if (errorDelta <= this.stallMinErrorDelta) {
        	
            this.consecutivelyStalledEpochCounter++;

            if (this.consecutivelyStalledEpochCounter >= this.maxConsecutivelyStalledEpochs) {
                return true;
            }
            
        } else {
        	
            this.consecutivelyStalledEpochCounter = 0;
            
        }

        return false;
    }
    
    public void resetStallTracking() {
    	
    	this.consecutivelyStalledEpochCounter = 0;
    	
    }

    protected double[] calculateOutputError(Vector desiredOutputVec, Vector currentOutputVec) {
        
    	double[] outputError = new double[ desiredOutputVec.size() ];
        
        for (int i = 0; i < currentOutputVec.size(); i++) {

        	outputError[ i ] = desiredOutputVec.get(i) - currentOutputVec.get(i);
        	
        }
        
        return outputError;
    }

    /**
     * Adjusted this to reflect a general RMS formula
     * 
     * @param outputError
     */
    protected void addToSquaredErrorSum(double[] outputError) {
        
    	double outputErrorSqrSum = 0.0d;
    	
        for (double error : outputError) {
        
        	//outputErrorSqrSum += (error * error) * 0.5d; // a;so multiply with 1/trainingSetSize  1/2n * (...)
        	outputErrorSqrSum += (error * error) * 0.5;
        	//System.out.println("> err^2 " + ((error * error)) );
        	
        }

        this.totalSquaredErrorSum += outputErrorSqrSum;
    }
    
    /**
     * Calculates Root Mean Square Error
     * - should be reset per each pass of a dataset!
     * 
     * @return
     */
    public double calcRMSError() {
    	//System.out.println("RMSECalc - Debug: " + this.totalSquaredErrorSum + " / " + ((double)this.recordsSeenDuringEpock) );
    	//System.out.println("Sqrt: " + Math.sqrt(this.totalSquaredErrorSum / ((double)this.recordsSeenDuringEpock)));
    
    	return Math.sqrt(this.totalSquaredErrorSum / ((double)this.recordsSeenDuringEpock));
    	
    }
    
    public boolean isInBatchMode() {
        return batchMode;
    }

    public void setBatchMode(boolean batchMode) {
        this.batchMode = batchMode;
    }    
    
    public void setRecordsSeen_Debug(long rec_seen) {
    	
    	this.recordsSeenDuringEpock = rec_seen;
    	
    }
    
    public void setLearningRate(double lr) {
    	
    	this.learningRate = lr;
    	
    }
    
	public boolean isStalled() {
		return false;
	}
	
	public void completeTrainingEpoch() {
		
		this.prevEpochTotalError = this.totalNetworkMeanSquaredError;
				
        this.totalNetworkMeanSquaredError = this.totalSquaredErrorSum / this.recordsSeenDuringEpock;

        
        // check for stall?
		
	}
    
    
    
    
}
