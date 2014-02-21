package tv.floe.metronome.deeplearning.neuralnetwork.core.learning;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;


public class TestAdagrad {

	@Test
	public void testCalcAdaptiveLearningRate() {
		
		Matrix gradient0 = new DenseMatrix(1, 1);
		gradient0.set( 0, 0, 1.0);

		Matrix gradient1 = new DenseMatrix(1, 1);
		gradient1.set( 0, 0, 1.0);
		
		Matrix gradient2 = new DenseMatrix(1, 1);
		gradient2.set( 0, 0, 2.0);

		
		AdagradLearningRate ada = new AdagradLearningRate(1, 1, 10);
		
		//ada.computeGradients();
		assertEquals(10, ada.getLearningRate(0, 0), 0.0 );
		//assertEquals(10, ada., 0.0 );
		
		ada.addLastIterationGradient( gradient0 );
		//ada.computeGradients();
		assertEquals(10, ada.getLearningRate(0, 0), 0.0 );
		
		ada.addLastIterationGradient( gradient1 );
		//ada.computeGradients();
		assertEquals(7.07, ada.getLearningRate(0, 0), 0.01 );
		
		ada.addLastIterationGradient( gradient2 );
		//ada.computeGradients();
		assertEquals(4.082, ada.getLearningRate(0, 0), 0.001 );
		
	}
	
}
