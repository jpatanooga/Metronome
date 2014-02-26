package tv.floe.metronome.deeplearning.rbm;

import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseNeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.core.NeuralNetworkGradient;
import tv.floe.metronome.deeplearning.neuralnetwork.optimize.NeuralNetworkOptimizer;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

/**
 * A Mallet-based optimizer for SGD in RBM
 * 
 * Current Status: currently there are some integration issues, this class is not functional
 * 
 * TODO: finish debugging
 * 
 * For now, use the stock SGD + CDk built into RBMs
 * 
 * @author josh
 *
 */
public class RestrictedBoltzmannMachineOptimizer extends NeuralNetworkOptimizer {

	
	private static final long serialVersionUID = 3676032651650426749L;
	protected int k = -1;
	protected int numTimesIterated = 0;
	
	public RestrictedBoltzmannMachineOptimizer(BaseNeuralNetworkVectorized network, double lr,
			Object[] trainingParams) {
		super(network, lr, trainingParams);
	}

	
	
	
	@Override
	public void getValueGradient(double[] buffer) {
		int k = (Integer) extraParams[0];
		numTimesIterated++;
		
		//System.out.println("k: " + k);
		
		//adaptive k based on the number of iterations.
		//typically over time, you want to increase k.
		if(this.k <= 0)
			this.k = k;
		if(numTimesIterated % 10 == 0) {
			this.k++;
		}
		
		
		//Don't go over 15
		if(this.k >= 15) 
		     this.k = 15;
		
		k = this.k;

		// this is where the RBM actually performs CDk
		NeuralNetworkGradient gradient = network.getGradient( new Object[]{ k, lr } );
		
		Matrix wAdd = gradient.getwGradient();
		Matrix vBiasAdd = gradient.getvBiasGradient();
		Matrix hBiasAdd = gradient.gethBiasGradient();
		
		int idx = 0;
		for (int i = 0; i < MatrixUtils.length( wAdd ); i++) { 
		
			buffer[idx++] = MatrixUtils.getElement( wAdd, i );
			
		}
		
		
		for (int i = 0; i < MatrixUtils.length( vBiasAdd ); i++) {
		
			buffer[idx++] = MatrixUtils.getElement( vBiasAdd, i );
			
		}
		

		
		for (int i = 0; i < MatrixUtils.length( hBiasAdd ); i++) {
			
			buffer[idx++] = MatrixUtils.getElement( hBiasAdd, i );
			
		}
				
		
	}
	
	
	
	

}