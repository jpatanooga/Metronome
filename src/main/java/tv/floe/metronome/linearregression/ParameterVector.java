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
	  
	  public int TrainedRecords = 0;
	  //public float AvgLogLikelihood = 0;
	  public float AvgError = 0;

		// ### The partial components needed to compute R-squared ###
		
		// partial sum of y (observations) for computing y-avg at master
		public double y_partial_sum = 0;
		public double y_avg = 0;
		//public long record_count = 0;
		public double SSyy_partial_sum = 0;
		public double SSE_partial_sum = 0;
		
	  
	  
	  public byte[] Serialize() throws IOException {
	    
	    // DataOutput d
	    
	    ByteArrayOutputStream out = new ByteArrayOutputStream();
	    DataOutput d = new DataOutputStream(out);
	    	    
	    d.writeDouble(this.y_partial_sum);
	    d.writeDouble(this.y_avg);
	    //d.writeLong(this.record_count);
	    d.writeDouble(this.SSyy_partial_sum);
	    d.writeDouble(this.SSE_partial_sum);
	    
	    d.writeInt(this.IterationComplete);
	    d.writeInt(this.CurrentIteration);
	    
	    d.writeInt(this.TrainedRecords);
	    //d.writeFloat(this.AvgLogLikelihood);
	    d.writeFloat(this.AvgError);
	    // buf.write
	    // MatrixWritable.writeMatrix(d, this.worker_gradient.getMatrix());
	    MatrixWritable.writeMatrix(d, this.parameter_vector);
	    // MatrixWritable.
	    
	    return out.toByteArray();
	  }
	  
	  public void Deserialize(byte[] bytes) throws IOException {
	    // DataInput in) throws IOException {
	    
	    ByteArrayInputStream b = new ByteArrayInputStream(bytes);
	    DataInput in = new DataInputStream(b);

		this.y_partial_sum = in.readDouble();
		this.y_avg = in.readDouble();
	//	this.record_count = in.readLong();
		this.SSyy_partial_sum = in.readDouble();
		this.SSE_partial_sum = in.readDouble();
	    
	    
	    this.IterationComplete = in.readInt();
	    this.CurrentIteration = in.readInt();
	    
	    this.TrainedRecords = in.readInt(); // d.writeInt(this.TrainedRecords);
	    //this.AvgLogLikelihood = in.readFloat(); // d.writeFloat(this.AvgLogLikelihood);
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
