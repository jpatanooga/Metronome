package tv.floe.metronome.classification.neuralnetworks.activation;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestSigmoidTransferFunction {

	// double input, expected, expected_derivative;
	double[][] test_rows = {
            {.1, .524979, .3493760},
            {.2, .549833, .347516},
            {.3, .574442, .344458},
            {.4, .598687, .340260},
            {.5, .622459, .335003},
            {.6, .645656, .328784},
            {.7, .668187, .321712},
            {.8, .6899744, .313909},
            {.9, .7109495, .305500}
        };
	
	

   @Test
    public void testGetOutput() {
	   
	   Sigmoid instance;
	    double input, expected, expected_derivative;

	    for (int x = 0; x < test_rows.length; x++ ) {

	    	input = test_rows[x][0];
	    	expected = test_rows[x][1];
	    	expected_derivative = test_rows[x][2];
	    	
	        instance = new Sigmoid();
	        instance.setSlope(1.0);
		   
		   
	        double result = instance.getOutput(input);
	        assertEquals(expected, result, 0.0001);    
	        
	        double output = instance.getOutput(input);
	        double result_deriv = instance.getDerivative(input);
	        assertEquals(expected_derivative, result_deriv, 0.00001);
	        
	    	
	    	
	    }
	    
    }
   
   @Test
   public void showRangeOfFunction() {
	   
	   Sigmoid instance;
	    double input, expected, expected_derivative;

        instance = new Sigmoid();
        instance.setSlope(1.0);

        double[] inputs = {
        		-11.0,
        		-8.0,
        		-4.0,
        		-3.0,
        		-2.0,
        		-1.0,
        		-0.9,
        		-0.8,
        		-0.7,
        		-0.6,
        		-0.5,
        		-0.4,
        		-0.3,
        		-0.2,
        		-0.1,
        		0.0,
        		0.1,
        		0.2,
        		0.3,
        		0.4,
        		0.5,
        		0.6,
        		0.7,
        		0.8,
        		0.9,
        		1.0,
        		1.1,
        		10
        };
        
	    for (int x = 0; x < inputs.length; x++ ) {

	    	input = inputs[x];
		   
		   
	        double result = instance.getOutput(input);

	        System.out.println("sigmod(" + input + "): " + result );
	    	
	    }	   
	   
   }


}
