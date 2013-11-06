package tv.floe.metronome.classification.neuralnetworks.learning.metrics;

import java.io.Serializable;

public class LearningAlgorithmMetrics implements Serializable {

	long weightUpdateOpCount = 0; 
	long errCalcOpCount = 0;
	long trainingTimeTotal = 0;
	long trainingRecords = 0;
	
	long currentRunTimerStartMS = 0;
	
	public void incWeightOpCount() {
		this.weightUpdateOpCount++;
	}
	
	public void incErrCalcOpCount() {
		this.errCalcOpCount++;
	}
	
	public void reset() {
		
		this.weightUpdateOpCount = 0;
		this.errCalcOpCount = 0;
		this.trainingRecords = 0;
		this.trainingTimeTotal = 0;
		
	}
	
	public void startTrainingRecordTimer() {
		
		this.currentRunTimerStartMS = System.currentTimeMillis();
		
	}
	
	public void stopTrainingRecordTimer() {
		long runTime = System.currentTimeMillis() - this.currentRunTimerStartMS;
		//System.out.println("runtTime: " + runTime);
		this.trainingTimeTotal += runTime;
		this.trainingRecords++;
	}
	
	public double calcAverageTrainTime() {
		
		if (0 == this.trainingRecords) {
			return 0;
		}
		
		return (double)this.trainingTimeTotal / (double)this.trainingRecords;
		
	}
	
/*	public void addTrainingRecordTiming(double time) {
		
		
	}
	*/
	public void PrintMetrics() {
		
		System.out.println("------------ Network Train Metrics -------------" );
		System.out.println("Weight Update Ops: " + this.weightUpdateOpCount);
		System.out.println("Err Calc Ops: " + this.errCalcOpCount);
		System.out.println("Avg Record Training Time: " + this.calcAverageTrainTime() + " (ms)" );
		System.out.println("------------ Network Train Metrics -------------" );
		
	}
	

}
