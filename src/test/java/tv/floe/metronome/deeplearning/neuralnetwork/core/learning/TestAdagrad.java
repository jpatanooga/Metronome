package tv.floe.metronome.deeplearning.neuralnetwork.core.learning;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;
import tv.floe.metronome.math.MatrixUtils;


public class TestAdagrad {

	@Test
	public void testCalcAdaptiveLearningRate() {
		
		Matrix gradient0 = new DenseMatrix(1, 1);
		gradient0.set( 0, 0, 1.0);

		Matrix gradient1 = new DenseMatrix(1, 1);
		gradient1.set( 0, 0, 1.0);
		
		Matrix gradient2 = new DenseMatrix(1, 1);
		gradient2.set( 0, 0, 0.0);

		
		AdagradLearningRate ada = new AdagradLearningRate(1, 1);
		Matrix out = ada.getLearningRates(gradient0);
		
//		MatrixUtils.debug_print( gradient0 );
//		MatrixUtils.debug_print( out );
		
		assertEquals( 0.01, out.get(0, 0), 0.0001);
		
		/*
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
		
		Matrix lrs = ada.getLearningRates();
		assertEquals(4.082, lrs.get(0, 0), 0.001 );
		*/
		
	}
	
	
	@Test
	public void testWriteLoadToStream() throws FileNotFoundException {
		
		String tmpFilename = "/tmp/adagrad.testWriteLoadToStream.model";
		
		AdagradLearningRate ada = new AdagradLearningRate(1, 1);
		
		
		FileOutputStream oFileOutStream = new FileOutputStream( tmpFilename, false);
		ada.write(oFileOutStream);
		
		AdagradLearningRate ada_deser = new AdagradLearningRate(2, 2);
		
		// read / load the model
		FileInputStream oFileInputStream = new FileInputStream( tmpFilename );
		ada_deser.load(oFileInputStream);
		
		
		assertEquals( ada.rows, ada_deser.rows );
		assertEquals( ada.cols, ada_deser.cols );

		MatrixUtils.assertSameLength(ada.adjustedGradient, ada_deser.adjustedGradient);
		
		MatrixUtils.assertSameLength(ada.historicalGradient, ada_deser.historicalGradient);
		
	}
	
}
