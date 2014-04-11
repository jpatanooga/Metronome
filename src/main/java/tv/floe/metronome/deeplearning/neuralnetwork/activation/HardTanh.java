package tv.floe.metronome.deeplearning.neuralnetwork.activation;

import org.apache.mahout.math.Matrix;

import tv.floe.metronome.math.MatrixUtils;


public class HardTanh implements ActivationFunction {

		/**
		 * 
		 */
		private static final long serialVersionUID = -8484119406683594852L;

		@Override
		public Matrix apply(Matrix matrix) {
			for(int i = 0; i < MatrixUtils.length( matrix ); i++) {
				double val = MatrixUtils.getElement(matrix, i); //matrix.get(i);
				if(val < -1 )
					val = -1;
				else if(val > 1)
					val = 1;
				else
					val = Math.tanh(val);
				//matrix.put(i,val);
				MatrixUtils.setElement(matrix, i, val);
			}
			
			return matrix;
		}

/*		@Override
		public DoubleMatrix applyDerivative(DoubleMatrix input) {
			for(int i = 0; i < input.length; i++) {
				double val = input.get(i);
				if(val < -1 )
					val = -1;
				else if(val > 1)
					val = 1;
				else
					val = 1 - Math.pow(Math.tanh(val),2);
				input.put(i,val);
			}
			
			return input;
			
		}
*/
		@Override
		public Matrix applyDerivative(Matrix input) {
			
			for (int i = 0; i < MatrixUtils.length( input ); i++) {
				
				double val = MatrixUtils.getElement(input, i); //input.get(i);
				if(val < -1 )
					val = -1;
				else if(val > 1)
					val = 1;
				else
					val = 1 - Math.pow(Math.tanh(val),2);
				//input.put(i,val);
				MatrixUtils.setElement(input, i, val);
				
			}
			
			return input;
		}	
	
}
