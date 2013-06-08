package tv.floe.metronome.vectorization;

import java.util.zip.CRC32;
import java.util.zip.Checksum;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import tv.floe.metronome.utils.Utils;

public class TestHashKernel {
	
	
	@Test
	public void testBasicHashKernel() {

		String line = "The brown cow ran down the street";

		HashKernel hk = new HashKernel(10);
		Vector v = hk.createCorrectlySizedVector();
		
		hk.hash(line, v);
		
		Utils.PrintVectorNonZero(v);
		
		v.assign(0);
		
		System.out.println( "Cleared: ");
		Utils.PrintVectorNonZero(v);
		
		
	}

}
