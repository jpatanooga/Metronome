package tv.floe.metronome.io.records;

import static org.junit.Assert.*;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class TestlibsvmRecordFactory {


	public void test() {
//		fail("Not yet implemented");
	}

	/**
	 * Data From:
	 * 
	 * http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a
	 * @throws Exception 
	 * 
	 * 
	 */
	@Test
	public void testParse() throws Exception {
		
		String record_0 = "-1 3:1 11:1 14:1 19:1 39:1 42:1 55:1 64:1 67:1 73:1 75:1 76:1 80:1 83:1 "; 
		String record_1 = "-1 3:1 6:1 17:1 27:1 35:1 40:1 57:1 63:1 69:1 73:1 74:1 76:1 81:1 103:1 ";
		String record_2 = "-1 4:1 6:1 15:1 21:1 35:1 40:1 57:1 63:1 67:1 73:1 74:1 77:1 80:1 83:1 ";
		
		
		libsvmRecordFactory rec_factory = new libsvmRecordFactory();
		
		Vector v = new RandomAccessSparseVector(RCV1RecordFactory.FEATURES);
		
		double actual_0 = rec_factory.processLineAlt(record_0, v);
		
		assertEquals( actual_0, -1.0, 0.0 );
		assertEquals( v.get(0), 1.0, 0.0 );
		assertEquals( v.get(4), 1.0, 0.0 );
		assertEquals( v.get(5), 0.0, 0.0 );
		assertEquals( v.get(12), 1.0, 0.0 );
		
		
		
		
		
	}
	
}
