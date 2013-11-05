package tv.floe.metronome.classification.neuralnetworks.learning.metrics;

import java.io.Serializable;

public class LearningAlgorithmMetrics implements Serializable {

	long weightUpdateOpCount = 0; 
	long errCalcOpCount = 0;
	
	public void incWeightOpCount() {
		this.weightUpdateOpCount++;
	}
	
	public void incErrCalcOpCount() {
		this.errCalcOpCount++;
	}
	
	public void reset() {
		
		this.weightUpdateOpCount = 0;
		this.errCalcOpCount = 0;
		
	}
	
	public void PrintMetrics() {
		
		System.out.println("Weight Update Ops: " + this.weightUpdateOpCount);
		System.out.println("Err Calc Ops: " + this.errCalcOpCount);
		
	}
	

}
