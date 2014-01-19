package tv.floe.metronome.deeplearning.rbm;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

public class TestRestrictedBoltzmannMachineOptimizer {

	public static Matrix buildTestInputDataset() {
		
		double[][] data = new double[][]
				{
					{1,1,1,0,0,0},
					{1,0,1,0,0,0},
					{1,1,1,0,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,0,0},
					{0,0,1,1,1,0},
					{0,0,1,1,1,0}
				};
		
		Matrix d = new DenseMatrix(data);

		return d;
		
	}	
	
	@Test
	public void testGetValueGradient() {

		Matrix input = buildTestInputDataset();
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 2, null);

		rbm.setInput(input);
		
		double[] buffer = new double[100];
		
		rbm.optimizer = new RestrictedBoltzmannMachineOptimizer(rbm, 0.1, new Object[]{1});
		rbm.optimizer.getValueGradient(buffer);
		
		
	}

}
