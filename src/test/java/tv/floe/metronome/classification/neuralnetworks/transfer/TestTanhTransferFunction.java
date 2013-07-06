package tv.floe.metronome.classification.neuralnetworks.transfer;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestTanhTransferFunction {

	
	double input,expected,expected_derivative;
	Tanh instance;
	  
	double[][] test_params = 	  
		{
	      {.1,.0249947929,.940014848},
	      {.2,.0499583,.786447},
	      {.3,.07485969,.596585},
	      {.4,.09966799,.419974},
	      {.5,.124353,.280414},
	      {.6,.1488850,.1807066},
	      {.7,.17323515,.113812},
	      {.8,.1973753,.070650},
	      {.9,.22127846,.043464},
	  };	
	
	
    @Test
    public void testGetOutput() {
        
	    for (int x = 0; x < test_params.length; x++ ) {

	    	input = test_params[x][0];
	    	expected = test_params[x][1];
	    	expected_derivative = test_params[x][2];
    	
	        instance=new Tanh();
	        instance.setSlope(0.5d);

	        
	        double result = instance.getOutput(input);
	        assertEquals(expected, result, 0.00001);
	        
	        double derv = instance.getDerivative(input);
	        
	        System.out.println("> expected: " + result);
	        
	        //result = instance.getOutput(input);
	        //assertEquals(expected,result,.0001);
	        //System.out.println("* expected: " + result);
	        
	        System.out.println( "curr derv " + derv );
	        System.out.println( "unit test answer derv: " + expected_derivative );
	        System.out.println( "tan^2: " + instance.getAltDerv(input) );
	        
	        
	    }
	    
    }

    /**
     * Test of getDerivative method, of class Tanh.
     */
    @Test
    public void testGetDerivative() {
        
 /*      
	    for (int x = 0; x < test_params.length; x++ ) {

	    	input = test_params[x][0];
	    	expected = test_params[x][1];
	    	expected_derivative = test_params[x][2];
    	
	        instance=new Tanh();
	        instance.setSlope(.5);

	    	
	        double out = instance.getOutput(input);
	        double derv = instance.getDerivative(input);
	        
	        System.out.println( "> " + derv );
	        System.out.println( "A " + expected_derivative );
	        
	        //assertEquals(expected_derivative, result, 0.00001);

	    }
      */ 
    }	
	

}
