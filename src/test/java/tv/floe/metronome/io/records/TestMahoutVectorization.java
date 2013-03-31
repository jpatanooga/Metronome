package tv.floe.metronome.io.records;

import static org.junit.Assert.*;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ContinuousValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.junit.Test;

import tv.floe.metronome.utils.Utils;

public class TestMahoutVectorization {

	@Test
	public void test() {
		
		
		FeatureVectorEncoder encoder = new ContinuousValueEncoder("demo");
		Vector v = new RandomAccessSparseVector( 3 );
		
		encoder.addToVector("32", 1, v);

		FeatureVectorEncoder encoder_other = new ContinuousValueEncoder("demo_other");
		//Vector v = new RandomAccessSparseVector( 3 );
		
		encoder_other.addToVector("100", 1, v);
		
		assertEquals( v.get(0), 132.0, 0.0);
		
		Utils.PrintVector(v);
		
	}

}
