package tv.floe.metronome.classification.neuralnetworks.activation;

/**
 * The hyperbolic tangent activation function
 * - produces output numbers between -1 and 1
 * - most useful for training data that is also between 0 and 1. 
 * - can be used with gradient descent based training methods.  
 * - common activation function used for neural networks
 * 
 * - provides similar scaling to the sigmoid activation function, 
 * --- the difference being: has a range from -1 to 1 
 * - often used in place of the sigmoid activation function.
 * 
 * 
 *
 */
public class Tanh  extends ActivationFunction {

	private double slope = 2d;

	public Tanh() {
	}

	public Tanh(double slope) {
		
		this.slope = slope;
		
	}



	@Override
	final public double getOutput(double net) {

		//System.out.println(">TANH!");
		
        if (net > 100) {
        
        	return 1.0;
        
        }else if (net < -100) {
        
        	return -1.0;
        
        }

		double E_x = Math.exp(this.slope * net);                
        
		this.output = (E_x - 1d) / (E_x + 1d);
                
		return this.output;                
	}
	
	@Override
	final public double getDerivative(double net) {
		
		//System.out.println( "> getDerv: " + this.output );
		
		return (1d - output * output);
		//return (1d - net * net);
		
	}	
	
	public double getAltDerv(double net) {
		
		return 1d - Math.pow( Math.tanh(this.output), 2d );
		
	}

	public double getSlope() {

		return this.slope;
		
	}

	public void setSlope(double slope) {

		this.slope = slope;
		
	}

}
