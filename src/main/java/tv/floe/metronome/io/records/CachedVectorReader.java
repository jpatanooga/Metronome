package tv.floe.metronome.io.records;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.io.Text;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.cloudera.iterativereduce.io.TextRecordParser;


/**
 * Used to cache in memory the records from the local block for uses in future passes.
 * 
 * 1. first pass reads from RecordReader and vectorizes, caches in internal data structure
 * 
 * 2. subsequent passes read from local cache
 * 
 * @author josh
 *
 */
public class CachedVectorReader {
		
	ArrayList<CachedVector> arCachedVectors = new ArrayList<CachedVector>();
	int currentVectorIndex = 0;
	
	TextRecordParser record_reader = null;
	RecordFactory vector_factory = null;
	boolean bCacheIsHot = false;
	
	
	public CachedVectorReader( TextRecordParser record_reader, RecordFactory vecFactory ) {
		
		this.record_reader = record_reader;
		this.vector_factory = vecFactory;
		
		
	}
	
	/**
	 * I spent too long of periods trying to finish this, it can probably be a lot better
	 * 
	 * @param cachedVec
	 * @return
	 * @throws IOException
	 */
	  public boolean next( CachedVector cachedVec ) throws IOException {
		    
		    Text value = new Text();
		    
		    boolean result = true;
		    
		    if ( this.bCacheIsHot ) {
		    	
		    	// cache is hot, read vector from there
		    	
		    	if ( this.currentVectorIndex >= this.arCachedVectors.size() ) {
		    		
		    		cachedVec.vec.assign(0.0);
		    		return false;
		    		
		    	} else {
		    	
		    		//System.out.println( "> hittin that cache: " + this.currentVectorIndex );
		    		cachedVec.vec.assign(this.arCachedVectors.get(this.currentVectorIndex).vec);
		    		cachedVec.label = this.arCachedVectors.get(this.currentVectorIndex).label;
			    	this.currentVectorIndex++;
			    	return true;
			    	
		    	}
		    	
		    	
		    	
		    } else {
	
			    if  (this.record_reader.hasMoreRecords()) {
			        
			    	// pull the value from the reader
			        try {
			          result = this.record_reader.next(value);
			        } catch (IOException e1) {
			          e1.printStackTrace();
			        }
			        
			        // vectorize the line
			        if (result) {
			        	
				      
				        //System.out.println( " value: " + value.toString() );
				        
				        CachedVector cVec = new CachedVector( this.vector_factory.getFeatureVectorSize() );
				        //cVec.vec = new RandomAccessSparseVector( this.vector_factory.getFeatureVectorSize() );
				        
				        try {
							cVec.label = this.vector_factory.processLineAlt(value.toString(), cVec.vec);
							//System.out.println("vec val: " + cVec.label);
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
				        
				        this.arCachedVectors.add(cVec);
				        cachedVec.vec.assign(cVec.vec);
				        cachedVec.label = cVec.label;
				        
				        
			        } else {
			        	
			        	// nothing to return, EOF
			        	// set cache hot
			        	this.bCacheIsHot = true;
			        	
			        }
	
			        return result;
			        
			    } else {
			        	
			    
			    	// flip this so next pass we read from the vector cache
		        	this.bCacheIsHot = true;
	
			    }
			    
		    } 
		        

		  
		  
		  return false;
	  }	
	  
	  public boolean hasMoreRecords() {
		  
		  return this.record_reader.hasMoreRecords();
	  }
	  
	  public void Reset() {
		  
		  this.currentVectorIndex = 0;
		  
	  }
	  
	  public boolean isCacheHot() {
		  
		  return this.bCacheIsHot;
		  
	  }
	

}
