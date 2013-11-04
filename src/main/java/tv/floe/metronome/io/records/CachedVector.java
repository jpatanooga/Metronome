package tv.floe.metronome.io.records;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

public class CachedVector {
	
	public Vector vec_input = null;
	public Vector vec_output = null;
	//public double label = 0.0;
	
	public CachedVector( int input_feature_vec_size, int output_vec_size ) {
		
		this.vec_input = new RandomAccessSparseVector( input_feature_vec_size );
		this.vec_output = new RandomAccessSparseVector( output_vec_size );
		
	}

}
