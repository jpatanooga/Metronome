package tv.floe.metronome.io.records;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;

/**
 * Built specifically for the Neural Network case
 * 
 * Parses schema similar to libsvm's format but takes a schema
 * schema:
 * 		i:N | o:M
 * Example:
 * 		i:10 | o:3
 * would parse
 * 		0:0.3 4:0.21 | 0:1.0 1:0.5 2:0.1
 * 
 * Parses sparse representations, but allows for definition of multiple output values
 * - pipe delimiter is left in to make it easier to work w the sparse representation
 * 
 * @author josh
 *
 */
public class MetronomeRecordFactory implements RecordFactory {
	  
	  public static final int FEATURES = 10000;
	  //ConstantValueEncoder encoder = null;
	  private boolean useBiasTerm = false;
	  private int featureVectorSize = FEATURES;
	  private String schema = "";
	  private int inputValues = 0;
	  private int outputValues = 0;
//	  double[] output_zeros; // = new double[this.outputValues];
//	  double[] input_zeros; 
	  
	  /**
	   * We'll let the feature vector size be set independently so we can do hashing tricks as needed
	   * 
	   * @param featureVectorSize
	   * @param schema
	   */
	  public MetronomeRecordFactory(String schema) {
	    
		  //this.featureVectorSize = featureVectorSize;
		  this.schema = schema;
	    //this.encoder = new ConstantValueEncoder("body_values");
	    this.parseSchema();
	  }
	  
	  public void setUseBiasTerm() {
		  this.useBiasTerm = true;
	  }
	  
	  public void parseSchema() {
		  
		  String[] parts = this.schema.split("\\|");
		  
		  //System.out.println("parts " + parts.length);
		  
		  for ( int x = 0; x < parts.length; x++) {
			  
			  //System.out.println("part " + x + ": " + parts[x]);

			  // now split schema into in/out parts
			  String[] type_vals = parts[x].trim().split(":");
			  if (type_vals[0].equals("i")) {
				  
				  // input vals
				  this.inputValues = Integer.parseInt(type_vals[1]);
//				  this.input_zeros = new double[this.inputValues];
				  this.featureVectorSize = this.inputValues;
				  
			  } else if (type_vals[0].equals("o")) {
				  
				  // output vals
				  this.outputValues = Integer.parseInt(type_vals[1]);
	//			  this.output_zeros = new double[this.outputValues];
				  
			  }
			  
		  }
		  
	  }
	  
	  @Override
	  public int getInputVectorSize() {
		  return this.inputValues;
	  }
	  
	  @Override
	  public int getOutputVectorSize() {
		  return this.outputValues;
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
		  throw new Exception("Use vectorizeLine!!!!");
	  }
	  
	  public void clearVector(Vector v) {
		  
		  Iterator<Element> it = v.iterateNonZero();
		  while (it.hasNext()) {
		    Element e = it.next();
		    e.set(0);
		  }
		  
	  }
	  
	  public void vectorizeLine(String line, Vector input_vec, Vector output_vec) {
	    
	    String[] inputs_outputs = line.split("\\|");
	    
	    //System.out.println("in: " + inputs_outputs[0]);
	    //System.out.println("out: " + inputs_outputs[1]);
	    
	    String[] inputs = inputs_outputs[0].trim().split(" ");
	    String[] outputs = inputs_outputs[1].trim().split(" ");
	    
	    //actual = Integer.parseInt(parts[0]);
	    //actual = Double.parseDouble(parts[0]);
	    
	    // clear it
	    input_vec.assign( 0.0 ); // this.input_zeros );
//	    input_vec.
//	    this.clearVector(input_vec);
	    
	    int startFeatureIndex = 0;
	    // dont know what to do the the "namespace" "f"
	    if (this.useBiasTerm) {
	    	input_vec.set(0, 1.0);
	    	startFeatureIndex = 1;
	    }
	    
	    for (int x = 0; x < inputs.length; x++) {
	      
	    	//System.out.println("> DEbug > part: " + parts[x]);
	    	
	      String[] feature = inputs[x].split(":");
	      
	      // get (offset) feature index and hash as necesary
	      int index = (Integer.parseInt(feature[0]) + startFeatureIndex) % this.featureVectorSize;
	      
	      double val = Double.parseDouble(feature[1]);
	      
//	      System.out.println(index + " -> " + feature[1] + " = " + val );
	      
	      if (index < this.featureVectorSize) {
	    	  input_vec.set(index, val);
	      } else {
	        
	        System.out
	            .println("Could Hash: " + index + " to " + (index % this.featureVectorSize));
	        
	      }
	      
	    }
	    
	    // zero out the output vec
	    this.clearVector(output_vec);
	    
	    for (int x = 0; x < outputs.length; x++) {
		      
	    	//System.out.println("> DEbug > part: " + outputs[x]);
	    	
	      String[] output_val = outputs[x].split(":");
	      
	      // get (offset) feature index and hash as necesary
	      int index = Integer.parseInt(output_val[0]);
	      
	      double val = Double.parseDouble(output_val[1]);

	      output_vec.set(index, val);

	    }
	    
	    // System.out.println("\nEOL\n");
	    
	    //return actual;
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
