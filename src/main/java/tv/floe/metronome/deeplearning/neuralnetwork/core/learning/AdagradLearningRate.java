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
	
	public AdagradLearningRate( int rows, int cols) {
		
		this.connectionLearningRates = new DenseMatrix(rows, cols);
		this.connectionLearningRates.assign(0.0);
		
		this.squaredGradientSums = new DenseMatrix(rows, cols);
		this.squaredGradientSums.assign(0.0);
		
	}
	
	public AdagradLearningRate(double gamma) {
		this.gamma = gamma;
	}
	
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
	
	/**
	 * eta == gamma / Math.sqrt( this.squaredGradientSum )
	 * 
	 * @return
	 */
	public void computeGradients() {
		//if ( this.squaredGradientSum > 0) {
			//return this.gamma / Math.sqrt( this.squaredGradientSum );
		//} else {
			//return this.gamma;
		//}
		
		for ( int r = 0; r < this.connectionLearningRates.numRows(); r++ ) {
			
			for ( int c = 0; c < this.connectionLearningRates.numCols(); c++ ) {
				
				this.connectionLearningRates.set( r, c, this.gamma / Math.sqrt( this.connectionLearningRates.get(r, c) ) );
				
			}
			
		}
		
	}

}
