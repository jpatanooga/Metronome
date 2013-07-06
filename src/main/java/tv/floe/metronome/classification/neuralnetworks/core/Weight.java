package tv.floe.metronome.classification.neuralnetworks.core;

public class Weight {


	public double value;
    public transient double weightChange;	
	
    
	public Weight() {
		this.value = Math.random() - 0.5d;
                this.weightChange = 0;
  	}

	public Weight(double value) {
		this.value = value;
 	}
        
	public void inc(double amount) {
		this.value += amount;
	}

	public void dec(double amount) {
		this.value -= amount;
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
