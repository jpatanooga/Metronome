package tv.floe.metronome.classification.neuralnetworks.learning.adagrad;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestAdagradLearningRate {

	@Test
	public void testDefault() {
		AdagradLearningRate ada = new AdagradLearningRate(10);
		
		assertEquals(10, ada.compute(), 0.0 );
		
	}

	@Test
	public void testCalcAdaptiveLearningRate() {
		
		AdagradLearningRate ada = new AdagradLearningRate(10);
		
		assertEquals(10, ada.compute(), 0.0 );
		
		ada.addLastIterationGradient(1.0);
		
		assertEquals(10, ada.compute(), 0.0 );
		
		ada.addLastIterationGradient(1.0);
		
		assertEquals(7.07, ada.compute(), 0.01 );
		
		ada.addLastIterationGradient(2.0);
		
		assertEquals(4.082, ada.compute(), 0.001 );
		
	}
	
	

}
