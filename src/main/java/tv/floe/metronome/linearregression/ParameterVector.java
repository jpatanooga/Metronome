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

	  // worker stuff to send out
//	  public int SrcWorkerPassCount = 0;
	  
	  public Matrix parameter_vector = null;
//	  public int GlobalPassCount = 0; // what pass should the worker dealing with?
	  
	  public int IterationComplete = 0; // 0 = no, 1 = yes
	  public int CurrentIteration = 0;
	  
	  public int TrainedRecords = 0;
	  public float AvgLogLikelihood = 0;
	  public float PercentCorrect = 0;
	  
	  public byte[] Serialize() throws IOException {
	    
	    // DataOutput d
	    
	    ByteArrayOutputStream out = new ByteArrayOutputStream();
	    DataOutput d = new DataOutputStream(out);
	    
	    // d.writeUTF(src_host);
//	    d.writeInt(this.SrcWorkerPassCount);
//	    d.writeInt(this.GlobalPassCount);
	    
	    d.writeInt(this.IterationComplete);
	    d.writeInt(this.CurrentIteration);
	    
	    d.writeInt(this.TrainedRecords);
	    d.writeFloat(this.AvgLogLikelihood);
	    d.writeFloat(this.PercentCorrect);
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
	    // this.src_host = in.readUTF();
//	    this.SrcWorkerPassCount = in.readInt();
//	    this.GlobalPassCount = in.readInt();
	    
	    this.IterationComplete = in.readInt();
	    this.CurrentIteration = in.readInt();
	    
	    this.TrainedRecords = in.readInt(); // d.writeInt(this.TrainedRecords);
	    this.AvgLogLikelihood = in.readFloat(); // d.writeFloat(this.AvgLogLikelihood);
	    this.PercentCorrect = in.readFloat(); // d.writeFloat(this.PercentCorrect);
	    
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
