package tv.floe.metronome.io.records;

import static org.junit.Assert.assertEquals;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class TestMetronomeVectorizatonFormat {

	@Test
	public void testParse() throws Exception {
		
		String schema = "i:200 | o:5";
		
		String record_0 = "3:1 11:1 14:1 19:1 39:1 42:1 55:1 64:1 67:1 73:1 75:1 76:1 80:1 83:1 | 0:1.0"; 
		String record_1 = "3:1 6:1 17:1 27:1 35:1 40:1 57:1 63:1 69:1 73:1 74:1 76:1 81:1 103:1 | 3:0.5 4:-1.0 ";
		
		
		MetronomeRecordFactory rec_factory = new MetronomeRecordFactory(schema);
		//rec_factory.setUseBiasTerm();
		
		assertEquals(200, rec_factory.getInputVectorSize() );
		assertEquals(5, rec_factory.getOutputVectorSize() );
		
		Vector v = new RandomAccessSparseVector(rec_factory.getFeatureVectorSize());
		v.set(0, 0.5);
		v.set(20, 0.5);
		v.set(40, 0.5);
		
		assertEquals(0.5d, v.get(20), 0.0d );
		
		rec_factory.clearVector(v);
		
		assertEquals(0.0d, v.get(0), 0.0d );
		assertEquals(0.0d, v.get(20), 0.0d );
		assertEquals(0.0d, v.get(40), 0.0d );
		
	
		//double actual_0 = rec_factory.processLineAlt(record_0, v);
		
		Vector v_in = new RandomAccessSparseVector(rec_factory.getFeatureVectorSize());
		Vector v_out = new RandomAccessSparseVector(rec_factory.getFeatureVectorSize());
		
		rec_factory.vectorizeLine(record_0, v_in, v_out);
		
		assertEquals( 0.0, v_in.get(0), 0.0 );
		assertEquals( 1.0, v_in.get(3), 0.0 );
		
		assertEquals( 1.0, v_out.get(0), 0.0 );

		
		
		
		
		
		
		Vector v_in_1 = new RandomAccessSparseVector(rec_factory.getFeatureVectorSize());
		Vector v_out_1 = new RandomAccessSparseVector(rec_factory.getOutputVectorSize());
		
		rec_factory.vectorizeLine(record_1, v_in_1, v_out_1);
		
		assertEquals( 1.0, v_in_1.get(3), 0.0 );
		assertEquals( 0.0, v_in_1.get(4), 0.0 );
		assertEquals( 1.0, v_in_1.get(6), 0.0 );
		
		assertEquals( 0.5, v_out_1.get(3), 0.0 );
		assertEquals( 0.0, v_out_1.get(2), 0.0 );
		assertEquals( -1.0, v_out_1.get(4), 0.0 );
		
		
		
		
/*		assertEquals( v.get(0), 1.0, 0.0 );
		assertEquals( v.get(4), 1.0, 0.0 );
		assertEquals( v.get(5), 0.0, 0.0 );
		assertEquals( v.get(12), 1.0, 0.0 );
	*/	
		
		
		
		
	}
	
	
	
	@Test
	public void testXorParse() throws Exception {
		
		/*
0 0:0 1:0
1 0:0 1:1
1 0:1 1:0
0 0:1 1:1

		 */
		
		String[] xor_recs = {
				"0:0 1:0 | 0:0",
				"0:0 1:1 | 0:1",
				"0:1 1:0 | 0:1",
				"0:1 1:1 | 0:0",
		};
		
		
		String schema = "i:2 | o:1";
		
		MetronomeRecordFactory rec_factory = new MetronomeRecordFactory( schema);
		
		assertEquals(rec_factory.getInputVectorSize(), 2);
		assertEquals(rec_factory.getOutputVectorSize(), 1);
		
		Vector v_in_0 = new RandomAccessSparseVector(rec_factory.getInputVectorSize());
		Vector v_out_0 = new RandomAccessSparseVector(rec_factory.getOutputVectorSize());

		rec_factory.vectorizeLine(xor_recs[0], v_in_0, v_out_0);
		
		Vector v_in_1 = new RandomAccessSparseVector(rec_factory.getInputVectorSize());
		Vector v_out_1 = new RandomAccessSparseVector(rec_factory.getOutputVectorSize());

		rec_factory.vectorizeLine(xor_recs[1], v_in_1, v_out_1);
		
		Vector v_in_2 = new RandomAccessSparseVector(rec_factory.getInputVectorSize());
		Vector v_out_2 = new RandomAccessSparseVector(rec_factory.getOutputVectorSize());

		rec_factory.vectorizeLine(xor_recs[2], v_in_2, v_out_2);
		
		Vector v_in_3 = new RandomAccessSparseVector(rec_factory.getInputVectorSize());
		Vector v_out_3 = new RandomAccessSparseVector(rec_factory.getOutputVectorSize());
		
		rec_factory.vectorizeLine(xor_recs[3], v_in_3, v_out_3);
		
		/*
		 * 
				"0:0 1:0 | 0:0",
				"0:0 1:1 | 0:1",
				"0:1 1:0 | 0:1",
				"0:1 1:1 | 0:0",

		 * 
		 */
		
		// rec 0: "0:0 1:0 | 0:0"
		assertEquals(0.0, v_in_0.get(0), 0.0);
		assertEquals(0.0, v_in_0.get(1), 0.0);
		assertEquals(0.0, v_out_0.get(0), 0.0);
		
		// rec 1: "0:0 1:1 | 0:1"
		assertEquals(0.0, v_in_1.get(0), 0.0);
		assertEquals(1.0, v_in_1.get(1), 0.0);
		assertEquals(1.0, v_out_1.get(0), 0.0);
		
		// rec 2: "0:1 1:0 | 0:1",
		assertEquals(1.0, v_in_2.get(0), 0.0);
		assertEquals(0.0, v_in_2.get(1), 0.0);
		assertEquals(1.0, v_out_2.get(0), 0.0);
		
		// rec 3: "0:1 1:1 | 0:0",
		assertEquals(1.0, v_in_3.get(0), 0.0);
		assertEquals(1.0, v_in_3.get(1), 0.0);
		assertEquals(0.0, v_out_3.get(0), 0.0);
		
		
		
		
		
	}	
	
	
	@Test
	public void testVectorAssignMechanics() {
		
		Vector v1 = new RandomAccessSparseVector(RCV1RecordFactory.FEATURES);

		v1.set(0, 1);
		v1.set(3, 1);

		System.out.println("orig: " + v1.get(0));
		System.out.println("orig: " + v1.get(3));

		
		Vector v_alt = v1.assign(0.0);
		
		System.out.println("assign: " + v1.get(0));
		System.out.println("assign: " + v1.get(3));
		
		System.out.println("assign-out: " + v_alt.get(0));
		System.out.println("assign-out: " + v_alt.get(3));
		
		
		
		
	}
	
	@Test
	public void testVectorizeIrisMetronomeFormatted() {
		
		String line = "0:0.64556962 1:0.795454545 2:0.202898551 3:0.08 | 0:1 1:0 2:0";
		
		String schema = "i:4 | o:3";
		
		MetronomeRecordFactory rec_factory = new MetronomeRecordFactory(schema);
		
		assertEquals(rec_factory.getInputVectorSize(), 4);
		assertEquals(rec_factory.getOutputVectorSize(), 3);
		
		Vector v_in_0 = new RandomAccessSparseVector(rec_factory.getInputVectorSize());
		Vector v_out_0 = new RandomAccessSparseVector(rec_factory.getOutputVectorSize());

		rec_factory.vectorizeLine( line, v_in_0, v_out_0 );
		
		assertEquals(0.795454545, v_in_0.get(1), 0.0);
		assertEquals(0.202898551, v_in_0.get(2), 0.0);
		
		// test output
		assertEquals(1.0, v_out_0.get(0), 0.0);
		assertEquals(0.0, v_out_0.get(1), 0.0);
		assertEquals(0.0, v_out_0.get(2), 0.0);
		
	}
	
}
