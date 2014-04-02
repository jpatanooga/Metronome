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


public class NeuralNetworkGradient {

	//private static final long serialVersionUID = 5611230066214840732L;
	private Matrix wGradient;
	private Matrix vBiasGradient;
	private Matrix hBiasGradient;
	
	public Matrix getwGradient() {
		return wGradient;
	}
	public void setwGradient(Matrix wGradient) {
		this.wGradient = wGradient;
	}
	public Matrix getvBiasGradient() {
		return vBiasGradient;
	}
	public void setvBiasGradient(Matrix vBiasGradient) {
		this.vBiasGradient = vBiasGradient;
	}
	public Matrix gethBiasGradient() {
		return hBiasGradient;
	}
	public void sethBiasGradient(Matrix hBiasGradient) {
		this.hBiasGradient = hBiasGradient;
	}
	public NeuralNetworkGradient(Matrix wGradient,
			Matrix vBiasGradient, Matrix hBiasGradient) {
		super();
		this.wGradient = wGradient;
		this.vBiasGradient = vBiasGradient;
		this.hBiasGradient = hBiasGradient;
	}
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result
				+ ((hBiasGradient == null) ? 0 : hBiasGradient.hashCode());
		result = prime * result
				+ ((vBiasGradient == null) ? 0 : vBiasGradient.hashCode());
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
		NeuralNetworkGradient other = (NeuralNetworkGradient) obj;
		if (hBiasGradient == null) {
			if (other.hBiasGradient != null)
				return false;
		} else if (!hBiasGradient.equals(other.hBiasGradient))
			return false;
		if (vBiasGradient == null) {
			if (other.vBiasGradient != null)
				return false;
		} else if (!vBiasGradient.equals(other.vBiasGradient))
			return false;
		if (wGradient == null) {
			if (other.wGradient != null)
				return false;
		} else if (!wGradient.equals(other.wGradient))
			return false;
		return true;
	}
	
	/**
	 * Serializes this to the output stream.
	 * @param os the output stream to write to
	 */
	public void write(OutputStream os) {
		try {

		    DataOutput d = new DataOutputStream(os);
			
		    MatrixWritable.writeMatrix(d, this.wGradient );
		    MatrixWritable.writeMatrix(d, this.hBiasGradient );
		    MatrixWritable.writeMatrix(d, this.vBiasGradient );			
		    

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

			this.wGradient = MatrixWritable.readMatrix( di );
			this.hBiasGradient = MatrixWritable.readMatrix( di );
			this.vBiasGradient = MatrixWritable.readMatrix( di );
			
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}		
	
	
	/**
	 * Adds the given gradient and this one together
	 * @param gradient the gradient to add
	 */
	public void add(NeuralNetworkGradient gradient) {
		
		//wGradient.addi(gradient.getwGradient());
		MatrixUtils.addi( this.wGradient, gradient.getwGradient() );
		//vBiasGradient.addi(gradient.getvBiasGradient());
		MatrixUtils.addi( this.vBiasGradient, gradient.getvBiasGradient() );
		//hBiasGradient.addi(gradient.gethBiasGradient());
		MatrixUtils.addi( this.hBiasGradient, gradient.gethBiasGradient() );
	}
	
	/**
	 * Divides the gradients by the given number
	 * @param num the number to divie by
	 */
	public void div(int num) {
		
		//wGradient.divi(num);
		MatrixUtils.divi(this.wGradient, num );
		
		//vBiasGradient.divi(num);
		MatrixUtils.divi(this.vBiasGradient, num );
		
		//hBiasGradient.divi(num);
		MatrixUtils.divi(this.hBiasGradient, num );
	}	
	
}
