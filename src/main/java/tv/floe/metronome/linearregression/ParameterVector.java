package tv.floe.metronome.linearregression;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;

public class ParameterVector {
	  
	  public Matrix parameter_vector = null;
	  
	  public int IterationComplete = 0; // 0 = no, 1 = yes
	  public int CurrentIteration = 0;
	  public long batchTimeMS = 0;
	  
	  public int TrainedRecords = 0;
	  public float AvgError = 0;

		// ### The partial components needed to compute R-squared ###
		
		// partial sum of y (observations) for computing y-avg at master
		public double y_partial_sum = 0;
		public double y_avg = 0;
		public double SSyy_partial_sum = 0;
		public double SSE_partial_sum = 0;
		
	  
	  
	  public byte[] Serialize() throws IOException {
	    
	    ByteArrayOutputStream out = new ByteArrayOutputStream();
	    DataOutput d = new DataOutputStream(out);
	    	    
	    d.writeDouble(this.y_partial_sum);
	    d.writeDouble(this.y_avg);
	    d.writeDouble(this.SSyy_partial_sum);
	    d.writeDouble(this.SSE_partial_sum);
	    
	    d.writeInt(this.IterationComplete);
	    d.writeInt(this.CurrentIteration);
	    d.writeLong(this.batchTimeMS);
	    d.writeInt(this.TrainedRecords);
	    d.writeFloat(this.AvgError);
	    MatrixWritable.writeMatrix(d, this.parameter_vector);
	    
	    return out.toByteArray();
	  }
	  
	  public void Deserialize(byte[] bytes) throws IOException {
	    
	    ByteArrayInputStream b = new ByteArrayInputStream(bytes);
	    DataInput in = new DataInputStream(b);

		this.y_partial_sum = in.readDouble();
		this.y_avg = in.readDouble();
		this.SSyy_partial_sum = in.readDouble();
		this.SSE_partial_sum = in.readDouble();
	    
	    
	    this.IterationComplete = in.readInt();
	    this.CurrentIteration = in.readInt();
	    this.batchTimeMS = in.readLong();
	    
	    this.TrainedRecords = in.readInt(); // d.writeInt(this.TrainedRecords);
	    this.AvgError = in.readFloat(); // d.writeFloat(this.PercentCorrect);
	    
	    this.parameter_vector = MatrixWritable.readMatrix(in);
	    
	  }
	  
	  public int numFeatures() {
	    return this.parameter_vector.numCols();
	  }
	  
	  public int numCategories() {
	    return this.parameter_vector.numRows();
	  }
	  
	  public void AccumulateVector( Vector vec ) {
		  
		  Vector v = this.parameter_vector.viewRow(0).plus(vec);
		  this.parameter_vector.viewRow(0).assign(v);
		  
	  }
	  
	  public void AverageVectors( int denominator ) {
		  
		  Vector v = this.parameter_vector.viewRow(0).divide(denominator);
		  this.parameter_vector.viewRow(0).assign(v);
		  
	  }
	  
}
