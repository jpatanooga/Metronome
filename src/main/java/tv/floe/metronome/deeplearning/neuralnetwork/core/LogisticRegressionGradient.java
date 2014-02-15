package tv.floe.metronome.deeplearning.neuralnetwork.core;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.io.Serializable;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;

import tv.floe.metronome.math.MatrixUtils;


public class LogisticRegressionGradient implements Serializable {

	
	private static final long serialVersionUID = -2843336269630396562L;
	private Matrix wGradient;
	private Matrix bGradient;
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result
				+ ((bGradient == null) ? 0 : bGradient.hashCode());
		result = prime * result
				+ ((wGradient == null) ? 0 : wGradient.hashCode());
		return result;
	}
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		LogisticRegressionGradient other = (LogisticRegressionGradient) obj;
		if (bGradient == null) {
			if (other.bGradient != null)
				return false;
		} else if (!bGradient.equals(other.bGradient))
			return false;
		if (wGradient == null) {
			if (other.wGradient != null)
				return false;
		} else if (!wGradient.equals(other.wGradient))
			return false;
		return true;
	}
	
	public LogisticRegressionGradient(Matrix wGradient,
			Matrix bGradient) {
		super();
		this.wGradient = wGradient;
		this.bGradient = bGradient;
	}
	
	public Matrix getwGradient() {
		return wGradient;
	}
	
	public void setwGradient(Matrix wGradient) {
		this.wGradient = wGradient;
	}
	
	public Matrix getbGradient() {
		//return bGradient.columnMeans();
		return MatrixUtils.columnMeans(bGradient);
	}
	
	public void setbGradient(Matrix bGradient) {
		this.bGradient = bGradient;
	}
	
	/**
	 * Serializes this to the output stream.
	 * @param os the output stream to write to
	 */
	public void write(OutputStream os) {
		try {

		    DataOutput d = new DataOutputStream(os);
			
		    MatrixWritable.writeMatrix(d, this.bGradient );
		    MatrixWritable.writeMatrix(d, this.wGradient );			
		    

		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}	
	
	/**
	 * Load (using {@link ObjectInputStream}
	 * @param is the input stream to load from (usually a file)
	 */
	public void load(InputStream is) {
		try {

			DataInput di = new DataInputStream(is);

			this.bGradient = MatrixWritable.readMatrix( di );
			this.wGradient = MatrixWritable.readMatrix( di );
			
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}		
	
}
