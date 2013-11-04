package tv.floe.metronome.io.records;

import static org.junit.Assert.assertEquals;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import com.cloudera.iterativereduce.io.TextRecordParser;

public class TestCachedVectorReader {
	
	private static int write_recs = 40000;
	
	private static String outputBaseDir = "/tmp/TestCachedVectorReader/";
	private static String outputFilename = "test_vectors.csv";

	String record_0 = "-1 3:1 11:1 14:1 19:1 39:1 42:1 55:1 64:1 67:1 73:1 75:1 76:1 80:1 83:1 "; 
	String record_1 = "-1 3:1 6:1 17:1 27:1 35:1 40:1 57:1 63:1 69:1 73:1 74:1 76:1 81:1 103:1 ";
	String record_2 = "-1 4:1 6:1 15:1 21:1 35:1 40:1 57:1 63:1 67:1 73:1 74:1 77:1 80:1 83:1 ";
	
	  private static JobConf defaultConf = new JobConf();
	  private static FileSystem localFs = null; 
	  static {
	    try {
	      defaultConf.set("fs.defaultFS", "file:///");
	      localFs = FileSystem.getLocal(defaultConf);
	    } catch (IOException e) {
	      throw new RuntimeException("init failure", e);
	    }
	  }
	
	
	private InputSplit[] generateDebugSplits(Path input_path, JobConf job) {

		long block_size = localFs.getDefaultBlockSize();

		System.out.println("default block size: " + (block_size / 1024 / 1024)
				+ "MB");

		// ---- set where we'll read the input files from -------------
		FileInputFormat.setInputPaths(job, input_path);

		// try splitting the file in a variety of sizes
		TextInputFormat format = new TextInputFormat();
		format.configure(job);

		int numSplits = 1;

		InputSplit[] splits = null;

		try {
			splits = format.getSplits(job, numSplits);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return splits;

	}
	
	
	public void writeVectorsToDisk() {
		
		
		
	    try {
	        
	        File base_dir = new File(outputBaseDir);
	        if (!base_dir.exists()) {
	          base_dir.mkdirs();
	        }
	        
	        File shard_file_0 = new File(outputBaseDir + outputFilename);
	        
	        if (shard_file_0.exists()) {
	          shard_file_0.delete();
	        }
	        
	        shard_file_0.createNewFile();
	        BufferedWriter shard_writer = new BufferedWriter(new FileWriter(shard_file_0));
	        
	        for (int x = 0; x < write_recs; x++ ) {
	        	
	        	shard_writer.write(record_0 + "\n");
	        	shard_writer.write(record_1 + "\n");
	        	shard_writer.write(record_2 + "\n" );
	        	
	        }
	        
	        shard_writer.flush();
	        shard_writer.close();
	        
	    } catch (Exception e) {
	    	e.printStackTrace();
	    }
		
	}
	
	
	@Test
	public void test() throws IOException {
		
		writeVectorsToDisk();
		
		

		libsvmRecordFactory rec_factory = new libsvmRecordFactory(libsvmRecordFactory.FEATURES);

	    JobConf job = new JobConf(defaultConf);
	    
	    Path workDir = new Path( outputBaseDir + outputFilename );

	    
	    
	    InputSplit[] splits = generateDebugSplits(workDir, job);
	    
	    System.out.println( "> splits: " + splits[0].toString() );

	    
	    TextRecordParser txt_reader = new TextRecordParser();

	    long len = Integer.parseInt(splits[0].toString().split(":")[2]
	        .split("\\+")[1]);

	    txt_reader.setFile(splits[0].toString().split(":")[1], 0, len);

//	    worker_model_builder.setRecordParser(txt_reader);
		
		
		
		CachedVectorReader cachedVecReader = new CachedVectorReader(txt_reader, rec_factory); 
		
		CachedVector cv = new CachedVector( rec_factory.getFeatureVectorSize(), 1 );
		
		int record_count = 0;
		
		long startMS = System.currentTimeMillis();
		
		while (cachedVecReader.next(cv)) {
			
			if (0 == record_count || 1 == record_count) {
				System.out.println("rec 0>  " + cv.vec_output.get(0) + " == " + cv.vec_input.toString() );
				assertEquals(-1.0, cv.vec_output.get(0), 0.0);
			}
			
			record_count++;
			
		}
		
		long totalNonCacheTime = System.currentTimeMillis() - startMS;
		
		System.out.println( "Non Cache Time: " + totalNonCacheTime + " ms" );
		
		assertEquals( 3 * write_recs, record_count );
		
		record_count = 0;

		cachedVecReader.Reset();
		
		System.out.println( "> cache is hot: " + cachedVecReader.isCacheHot() );
		
		
		startMS = System.currentTimeMillis();
		
		while (cachedVecReader.next(cv)) {
			
			
			record_count++;
			
		}
		
		long totalCacheTime = System.currentTimeMillis() - startMS;
		
		System.out.println( "Cache Time: " + totalCacheTime + " ms" );
		
		assertEquals( 3 * write_recs, record_count );

		
		
		
		
		cachedVecReader.Reset();
		startMS = System.currentTimeMillis();
		
		while (cachedVecReader.next(cv)) {
			
			
			record_count++;
			
		}
		
		totalCacheTime = System.currentTimeMillis() - startMS;
		
		System.out.println( "Cache Time: " + totalCacheTime + " ms" );
		
		cachedVecReader.Reset();
		startMS = System.currentTimeMillis();
		
		while (cachedVecReader.next(cv)) {
			
			
			record_count++;
			
		}
		
		totalCacheTime = System.currentTimeMillis() - startMS;
		
		System.out.println( "Cache Time: " + totalCacheTime + " ms" );
				
		
//	    double actual = factory.processLineAlt(training_rec_0, v);
	    
//	    assertEquals( 0.657, actual, 0.001 );
		
		
		
	}
	
	
}
