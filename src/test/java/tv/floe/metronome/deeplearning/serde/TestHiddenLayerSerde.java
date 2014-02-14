package tv.floe.metronome.deeplearning.serde;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

import tv.floe.metronome.deeplearning.neuralnetwork.layer.HiddenLayer;
import tv.floe.metronome.math.MatrixUtils;

public class TestHiddenLayerSerde {

	@Test
	public void test() {
		
		Matrix input = new DenseMatrix(new double[][] 
		{
				{1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
				,{1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
				,{1,1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0}
				,{1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
				,{0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0}
				,{0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1}
				,{0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1}
				,{0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,0,1}
				,{0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1}
				,{0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0}
        }
		);
		
		MatrixUtils.debug_print_matrix_stats( input, "input" );
		
		
		
		
		
		RandomGenerator r = new MersenneTwister( 1234 );
		
		HiddenLayer layer = new HiddenLayer( 20, 2, r );
		layer.setInput( input );
		
		
		
	}

}
