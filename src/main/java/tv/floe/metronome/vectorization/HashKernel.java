package tv.floe.metronome.vectorization;

import java.util.zip.CRC32;
import java.util.zip.Checksum;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * Basic HashKernel based on:
 * 
 * 	http://jeremydhoon.github.com/2013/03/19/abusing-hash-kernels-for-wildly-unprincipled-machine-learning/
 * 
 * 
 * 
 * @author josh
 *
 */
public class HashKernel {

	private int bits = 8; // just a default
	
	public HashKernel( int bits ) {
		
		this.bits = bits;
		
	}
	
	
	/*

		feature_vector = [0 for _ in xrange(1 << bits)]
		for word in input_text.split():
		  hash = crc32(word)  # CRC-32 hash function, available in the zlib module
		  index = hash & ((1 << bits) - 1)
		  # Use the nth bit, zero-indexed, to determine if we add or subtract
		  # one from the index.
		  sign = (((hash & (1 << bits)) >> bits) << 1) - 1
		  feature_vector[index] += sign

	 * 
	 */

	private static long getCRC32(String word) {
		
        byte bytes[] = word.getBytes();
        
        Checksum checksum = new CRC32();
       
        /*
         * To compute the CRC32 checksum for byte array, use
         *
         * void update(bytes[] b, int start, int length)
         * method of CRC32 class.
         */
         
        checksum.update(bytes,0,bytes.length);
       
        /*
         * Get the generated checksum using
         * getValue method of CRC32 class.
         */
        long lngChecksum = checksum.getValue();
       
        //System.out.println("CRC32 checksum for byte array is :" + lngChecksum);		
		
        return lngChecksum;
        
	}
	
	public Vector createCorrectlySizedVector() {

		int vector_size = 1 << this.bits;

		Vector v = new RandomAccessSparseVector(vector_size);

		return v;
		
	}
	
	
	public void hash(String text, Vector v) {
		

		//int bits = 13;
		int vector_size = 1 << bits;

		v.assign(0);
		
		
		System.out.println( "vec size: " + vector_size );
		
        String[] parts = text.split(" ");

        for ( int x = 0; x < parts.length; x++ ) { 
        	
        	// 1.	CRC32 hash the word
        	
        	long hash = getCRC32(parts[x]); // crc something something here
        	
        	// 2.	create the index number
        	// index = hash & ((1 << bits) - 1)
        	int index = (int) (hash & ((1 << bits) - 1));
        	
        	//  # Use the nth bit, zero-indexed, to determine if we add or subtract
        	//  # one from the index.
        	//  sign = (((hash & (1 << bits)) >> bits) << 1) - 1        	
  		  	long sign = (((hash & (1 << bits)) >> bits) << 1) - 1;
  		  	
  		  	System.out.println( "> " + index + " : " + sign );
  		  	
  			v.set(index, v.get(index) + sign );
        	
        	
        }	
        
        
		
	}
	
	
	
}
