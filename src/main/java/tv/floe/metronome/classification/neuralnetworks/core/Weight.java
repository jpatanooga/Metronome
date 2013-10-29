package tv.floe.metronome.classification.neuralnetworks.core;

import java.io.Serializable;

public class Weight implements Serializable {


	public double value;
    public transient double weightChange;	
	
    
	public Weight() {
		this.value = Math.random() - 0.5d;
                this.weightChange = 0;
  	}

	public Weight(double value) {
		this.value = value;
 	}
        
	public void accumulate(double amount) {
		this.value += amount;
		//System.out.println("> Accum (+ " + amount + " )  : " + this.value);
	}

	public void decreaseBy(double amount) {
		this.value -= amount;
	}
	
	public void average(double denominator) {
		this.value = this.value / denominator;
	}

	public void setValue(double value) {
		this.value = value;
	}

	public double getValue() {
		return this.value;
	}
	
	public void randomize() {
		this.value = Math.random() - 0.5d;
	}

	public void randomize(double min, double max) {
            this.value = min + Math.random() * (max - min);
	}
    
    
}
