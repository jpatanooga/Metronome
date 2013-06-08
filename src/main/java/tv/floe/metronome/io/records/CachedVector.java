package tv.floe.metronome.io.records;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

public class CachedVector {
	
	public Vector vec = null;
	public double label = 0.0;
	
	public CachedVector( int feature_vec_size ) {
		
		this.vec = new RandomAccessSparseVector( feature_vec_size );
		
	}

}
