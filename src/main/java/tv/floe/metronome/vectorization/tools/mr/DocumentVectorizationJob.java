package tv.floe.metronome.vectorization.tools.mr;

import java.io.IOException;
import java.util.Iterator;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

/**
 * Simple MapReduce job to convert a directory of documents into 
 * a libSVM format vectorized dataset for use with Metronome
 * 
 * Reducer in this case is only really used for grouping records into 
 * a particular file size
 * 
 * TODO:
 * 
 * - copy MAHOUT-833's grouped document input to process a whole document 
 * per record w the whole record reader
 * 
 * - configuration parameters
 * 		-	number of records per output file (how large of input files do we want to for the ML job?)
 * 		-	
 * 
 * @author josh
 *
 */
public class DocumentVectorizationJob {

	
	
	  public static void main(String[] args) throws IOException {
	    JobConf conf = new JobConf(DocumentVectorizationJob.class);
	    
	    Configuration conf_alt = new Configuration();
	    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
	    
	   // System.out.println();
	    
	    
	        if (otherArgs.length != 2) {
	          System.err.println("Usage: DocumentVectorizationJob <in> <out>");
	          System.exit(2);
	        } else {
	        	System.out.println("Conf good!");
	        }
	    
	    conf.setJobName("DocumentVectorizationJob");

	    conf.setMapperClass(IdentityMapper.class);
	    conf.setReducerClass(IdentityReducer.class);
	    
	    conf.setMapOutputKeyClass(Text.class);
	      conf.setMapOutputValueClass(Text.class);

	      conf.setOutputKeyClass(NullWritable.class);
	      conf.setOutputValueClass(Text.class);

	      
	      conf.setInputFormat(TextInputFormat.class);
	      conf.setOutputFormat(TextOutputFormat.class);	      

	    if (otherArgs.length < 2) {
	      System.out.println("ERROR: Wrong number of parameters");
	      System.out.println("trivial <input_path> <output_path>");
	      System.exit(1);
	    }

	    
	    //conf.setInputPath(new Path(args[0]));
	    //conf.setOutputPath(new Path(args[1]));
	    
	    FileInputFormat.setInputPaths(conf, new Path(otherArgs[0]));
	      FileOutputFormat.setOutputPath(conf, new Path(otherArgs[1]));	    

	    JobClient.runJob(conf);
	  }

	  
	  
	  
	  
	  
	  /**
	   * 1. Take a complete file as the K/V pair per map call
	   * 
	   * 2. convert document into libSVM output record
	   * 
	   * 
	   * 
	   * @author josh
	   *
	   */
	  public static class IdentityMapper extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text> {
		  Text w = new Text();
		  Text v = new Text("1");
		  
		  long configure_start_time = 0;
		  long close_end_time = 0;
		  
		  long rec_count = 0;
		  
		  public void configure( JobConf c ) {
			  
			  this.configure_start_time = System.currentTimeMillis();
			  
		  }
		  
		  public void close() {
			  
			  
			  this.close_end_time = System.currentTimeMillis();
			  
			  long total_time = this.close_end_time - this.configure_start_time;
			  
			  System.out.println( ">>>>> Total Map Task Time: " + total_time );
			  
		  }
		  
	      public void map(LongWritable key, Text val, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
	    	  //k.set(key.toString());
/*	    	  StringTokenizer itr = new StringTokenizer(val.toString());
	          //output.collect(k, val);
	    	  while (itr.hasMoreTokens()) {
	    		   w.set(itr.nextToken());
*/	    		   
	    	  long key_id = (rec_count % 4);
	    	  
	    	  w.set(Long.toString(key_id));
	    	    
	    	  output.collect(w, val); 	  
	    	  
	        }
	  }

	  public static class IdentityReducer extends MapReduceBase implements Reducer<Text, Text, NullWritable, Text> {
		  public void reduce(Text key, Iterator<Text> values,
			      OutputCollector<NullWritable, Text> output, Reporter reporter) throws IOException {
			    //Set<String> attackers = new TreeSet<String>();
			  NullWritable nw = NullWritable.get();
			  
			    while (values.hasNext()) {
			      //String valStr = values.next().toString();
			      //attackers.add(valStr);
			    	output.collect(nw, new Text(values.next().toString()));
			    }
			    
			  }
			}
}