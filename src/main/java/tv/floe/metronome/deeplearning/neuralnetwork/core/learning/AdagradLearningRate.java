package tv.floe.metronome.deeplearning.neuralnetwork.core.learning;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

import tv.floe.metronome.math.MathUtils;
import tv.floe.metronome.math.MatrixUtils;

/**
 * 
 * Vectorized Learning Rate used per layer
 * 
 * @author josh
 *
 */
public class AdagradLearningRate {

	private double gamma = 10; // default for gamma (this is the numerator)
	//private double squaredGradientSum = 0;
	public Matrix squaredGradientSums;
	public Matrix connectionLearningRates;
	
	public AdagradLearningRate( int rows, int cols, double gamma) {
		
		this.connectionLearningRates = new DenseMatrix(rows, cols);
		this.connectionLearningRates.assign(0.0);
		
		this.squaredGradientSums = new DenseMatrix(rows, cols);
		this.squaredGradientSums.assign(0.0);

		this.gamma = gamma;

	}
	
//	public AdagradLearningRate(double gamma) {
//	}
	
	/**
	 * square gradient, then add to ongoing sum
	 * 
	 * @param gradient
	 */
	public void addLastIterationGradient(Matrix gradient) {
		
		//this.squaredGradientSum += gradient * gradient;
		Matrix gradientsSquared = MatrixUtils.pow(gradient, 2);
		
		//this.connectionLearningRates.
		MatrixUtils.addi(this.squaredGradientSums, gradientsSquared);
		
	}
	
	
	public double getLearningRate(int row, int col) {
		
		double squaredGradientSum = this.squaredGradientSums.get(row, col);
		
		if ( squaredGradientSum > 0) {
			return this.gamma / Math.sqrt( squaredGradientSum );
		} else {
			return this.gamma;
		}
		
		
	}

	public Matrix getLearningRates() {
		
		for ( int r = 0; r < this.squaredGradientSums.numRows(); r++ ) {
			
			for ( int c = 0; c < this.squaredGradientSums.numCols(); c++ ) {
				
				double squaredGradientSum = this.squaredGradientSums.get(r, c);
				
				if ( squaredGradientSum > 0) {
					this.connectionLearningRates.set(r, c, this.gamma / Math.sqrt( squaredGradientSum ) );
				} else {
					//this.gamma;
					this.connectionLearningRates.set(r, c, this.gamma);
				}
				
			}
			
		}
		
		return this.connectionLearningRates;
		
		
	}
	
	
}
