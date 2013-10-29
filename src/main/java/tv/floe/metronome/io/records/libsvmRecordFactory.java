package tv.floe.metronome.io.records;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;

/**
 * Record Factory for LibSVM
 * 
 * [class/target] 1:[firstFeatureValue] 2:[secondFeatureValue] etc.
 * 
 * Datasets
 * 
 * http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
 * 
 * QUESTIONS
 * 
 * - how are intercepts handled with libSVM?
 * 
 * 
 * @author josh
 *
 */
public class libsvmRecordFactory implements RecordFactory {
	  
	  public static final int FEATURES = 10000;
	  //ConstantValueEncoder encoder = null;
	  private boolean useBiasTerm = false;
	  private int featureVectorSize = FEATURES;
	  
	  public libsvmRecordFactory(int featureVectorSize) {
	    
		  this.featureVectorSize = featureVectorSize;
		  
	    //this.encoder = new ConstantValueEncoder("body_values");
	    
	  }
	  
	  public void setUseBiasTerm() {
		  this.useBiasTerm = true;
	  }
	  
	  public static void ScanFile(String file, int debug_break_cnt)
	      throws IOException {
	    
	    //ConstantValueEncoder encoder_test = new ConstantValueEncoder("test");
	    
	    BufferedReader reader = null;
	    // Collection<String> words
	    int line_count = 0;
	    
	    Multiset<String> class_count = ConcurrentHashMultiset.create();
	    Multiset<String> namespaces = ConcurrentHashMultiset.create();
	    
	    try {
	      // System.out.println( newsgroup );
	      reader = new BufferedReader(new FileReader(file));
	      
	      String line = reader.readLine();
	      
	      while (line != null && line.length() > 0) {
	        
	        // shard_writer.write(line + "\n");
	        // out += line;
	        
	    	  
	    	  
	        String[] parts = line.split(" ");
	        
	        // System.out.println( "Class: " + parts[0] );
	        
	        class_count.add(parts[0]);
	        //namespaces.add(parts[1]);
	        
	        line = reader.readLine();
	        line_count++;
	        
	        Vector v = new RandomAccessSparseVector(FEATURES);
	        
	        for (int x = 1; x < parts.length; x++) {
	          // encoder_test.addToVector(parts[x], v);
	          // System.out.println( parts[x] );
	          String[] feature = parts[x].split(":");
	          int index = Integer.parseInt(feature[0]) % FEATURES;
	          double val = Double.parseDouble(feature[1]);
	          
	          System.out.println( feature[1] + " = " + val );
	          
	          if (index < FEATURES) {
	            v.set(index, val);
	          } else {
	            
	            System.out.println("Could Hash: " + index + " to "
	                + (index % FEATURES));
	            
	          }
	          
	        }
	        
//	        Utils.PrintVectorSectionNonZero(v, 10);
	        System.out.println("###");
	        
	        if (line_count > debug_break_cnt) {
	          break;
	        }
	        
	      }
	      
	      System.out.println("Total Rec Count: " + line_count);
	      
	      System.out.println("-------------------- ");
	      
	      System.out.println("Classes");
	      for (String word : class_count.elementSet()) {
	        System.out.println("Class " + word + ": " + class_count.count(word)
	            + " ");
	      }
	      
	      System.out.println("-------------------- ");
	      
	      System.out.println("NameSpaces:");
	      for (String word : namespaces.elementSet()) {
	        System.out.println("Namespace " + word + ": " + namespaces.count(word)
	            + " ");
	      }
	      
	      /*
	       * TokenStream ts = analyzer.tokenStream("text", reader);
	       * ts.addAttribute(CharTermAttribute.class);
	       * 
	       * // for each word in the stream, minus non-word stuff, add word to
	       * collection while (ts.incrementToken()) { String s =
	       * ts.getAttribute(CharTermAttribute.class).toString();
	       * //System.out.print( " " + s ); //words.add(s); out += s + " "; }
	       */

	    } finally {
	      reader.close();
	    }
	    
	    // return out + "\n";
	    
	  }
	  
	  // doesnt really do anything in a 2 class dataset
	  @Override
	  public String GetClassnameByID(int id) {
	    return String.valueOf(id); // this.newsGroups.values().get(id);
	  }
	  
	  /**
	   * Processes single line of input into: - target variable - Feature vector
	   * 
	   * Right now our hash function is simply "modulo"
	   * 
	   * @throws Exception
	   */
	  @Override
	  public double processLineAlt(String line, Vector v) throws Exception {
	    
	    // p.269 ---------------------------------------------------------
	    // Map<String, Set<Integer>> traceDictionary = new TreeMap<String,
	    // Set<Integer>>();
	    
	    double actual = 0;
	    
	    //System.out.println("Line: " + line);
	    
	    String[] parts = line.split(" ");
	    
	    //actual = Integer.parseInt(parts[0]);
	    actual = Double.parseDouble(parts[0]);
	    
	    int startFeatureIndex = 0;
	    // dont know what to do the the "namespace" "f"
	    if (this.useBiasTerm) {
	    	v.set(0, 1.0);
	    	startFeatureIndex = 1;
	    }
	    
	    for (int x = 1; x < parts.length; x++) {
	      
	    	//System.out.println("> DEbug > part: " + parts[x]);
	    	
	      String[] feature = parts[x].split(":");
	      int index = (Integer.parseInt(feature[0]) + startFeatureIndex) % this.featureVectorSize;
	      double val = Double.parseDouble(feature[1]);
	      
//	      System.out.println(index + " -> " + feature[1] + " = " + val );
	      
	      if (index < this.featureVectorSize) {
	        v.set(index, val);
	      } else {
	        
	        System.out
	            .println("Could Hash: " + index + " to " + (index % this.featureVectorSize));
	        
	      }
	      
	    }
	    
	    // System.out.println("\nEOL\n");
	    
	    return actual;
	  }
	  
	  @Override
	  public List<String> getTargetCategories() {
	    
	    List<String> out = new ArrayList<String>();
	    
	    // for ( int x = 0; x < this.newsGroups.size(); x++ ) {
	    
	    // System.out.println( x + "" + this.newsGroups.values().get(x) );
	    out.add("0");
	    out.add("1");
	    
	    // }
	    
	    return out;
	    
	  }

	@Override
	public int processLine(String line, Vector featureVector) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getFeatureVectorSize() {
		return this.featureVectorSize;
	}
	
	
}
