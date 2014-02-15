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

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tv.floe.metronome.math.MatrixUtils;


public class LogisticRegression implements Serializable {

	private static final long serialVersionUID = -7065564817460914364L;
	public int nIn;
	public int nOut;
	public Matrix input,labels;
	public Matrix connectionWeights;
	public Matrix biasTerms;
	public double l2 = 0.01;
	public boolean useRegularization = true;
	private static Logger log = LoggerFactory.getLogger(LogisticRegression.class);

	// used for Serde
	public LogisticRegression() {}

	public LogisticRegression(Matrix input,Matrix labels, int nIn, int nOut) {
		
		this.input = input;
		this.labels = labels;
		this.nIn = nIn;
		this.nOut = nOut;
		//this.connectionWeights = Matrix.zeros(nIn,nOut);
		this.connectionWeights = new DenseMatrix(nIn, nOut);
		this.connectionWeights.assign(0.0);
		this.biasTerms = new DenseMatrix(1, nOut); //Matrix.zeros(nOut);
		this.biasTerms.assign(0.0);
		
	}

	public LogisticRegression(Matrix input, int nIn, int nOut) {
		
		this(input,null,nIn,nOut);
		
	}

	public LogisticRegression(int nIn, int nOut) {
		
		this(null,null,nIn,nOut);
		
	}

	public void train(double lr) {
		
		train(input,labels,lr);
		
	}


	public void train(Matrix x,double lr) {
		
		train(x,labels,lr);

	}

	public void merge(LogisticRegression l,int batchSize) {
		
		//W.addi(l.W.subi(W).div(batchSize));
		this.connectionWeights.plus(l.connectionWeights.minus(this.connectionWeights).divide(batchSize));
		
		//b.addi(l.b.subi(b).div(batchSize));
		this.biasTerms.plus(l.biasTerms.minus(this.biasTerms).divide(batchSize));
	}

	/**
	 * Objective function:  minimize negative log likelihood
	 * @return the negative log likelihood of the model
	 */
	public double negativeLogLikelihood() {
		
		//Matrix sigAct = softmax(input.mmul(W).addRowVector(b));
		Matrix sigActivation = MatrixUtils.softmax( MatrixUtils.addRowVector( input.times(this.connectionWeights), this.biasTerms.viewRow(0) ) );
		
		if (this.useRegularization) {
			
			//double reg = (2 / l2) * MatrixFunctions.pow(this.W,2).sum();
			// TODO: fix this
			double regularization = ( 2 / l2 ) * MatrixUtils.sum( MatrixUtils.pow(this.connectionWeights, 2) );
					
					
/*			return - labels.mul(log(sigAct)).add(
					oneMinus(labels).mul(
							log(oneMinus(sigAct))
							))
							.columnSums().mean() + reg;
*/
			//return - MatrixUtils.mean( MatrixUtils.columnSums( labels.times(MatrixUtils.log(sigActivation).plus( MatrixUtils.oneMinus(labels).times(MatrixUtils.log(MatrixUtils.oneMinus(sigActivation))) ) ) ) ) + regularization;
			
			Matrix labelsMulLogSig = MatrixUtils.elementWiseMultiplication( labels, MatrixUtils.log(sigActivation) );

			Matrix oneMinusLabelsMulLogOneMinusSigAct = MatrixUtils.elementWiseMultiplication( MatrixUtils.oneMinus(labels), MatrixUtils.log(MatrixUtils.oneMinus(sigActivation)));
			
			Matrix sum = labelsMulLogSig.plus( oneMinusLabelsMulLogOneMinusSigAct );
					
			return - MatrixUtils.mean( MatrixUtils.columnSums( sum ) )  + regularization;
			
					
					
		}
		/*
		return - labels.mul(log(sigAct)).add(
				oneMinus(labels).mul(
						log(oneMinus(sigAct))
						))
						.columnSums().mean();
*/
		
//		return - MatrixUtils.mean( MatrixUtils.columnSums( MatrixUtils.elementWiseMultiplication( labels, MatrixUtils.log(sigActivation) ).plus( MatrixUtils.oneMinus(labels).times(MatrixUtils.log(MatrixUtils.oneMinus(sigActivation))) ) ) ) );return - MatrixUtils.mean( MatrixUtils.columnSums( MatrixUtils.elementWiseMultiplication( labels, MatrixUtils.log(sigActivation) ).plus( MatrixUtils.oneMinus(labels).times(MatrixUtils.log(MatrixUtils.oneMinus(sigActivation))) ) ) ) );

		Matrix labelsMulLogSig = MatrixUtils.elementWiseMultiplication( labels, MatrixUtils.log(sigActivation) );

		Matrix oneMinusLabelsMulLogOneMinusSigAct = MatrixUtils.elementWiseMultiplication( MatrixUtils.oneMinus(labels), MatrixUtils.log(MatrixUtils.oneMinus(sigActivation)));
		
		Matrix sum = labelsMulLogSig.plus( oneMinusLabelsMulLogOneMinusSigAct );
				
		return - MatrixUtils.mean( MatrixUtils.columnSums( sum ) );
		
		
	}

	/**
	 * Train on the given inputs and labels.
	 * This will assign the passed in values
	 * as fields to this logistic function for 
	 * caching.
	 * @param x the inputs to train on
	 * @param y the labels to train on
	 * @param lr the learning rate
	 */
	public void train(Matrix x, Matrix y, double lr) {
		//ensureValidOutcomeMatrix(y);
		MatrixUtils.ensureValidOutcomeMatrix(y);
		
		if (x.numRows() != y.numRows()) {
			throw new IllegalArgumentException("How does this happen?");
		}

		this.input = x;
		this.labels = y;

		LogisticRegressionGradient gradient = getGradient(lr);

		//W.addi(gradient.getwGradient());
		this.connectionWeights.plus(gradient.getwGradient());
		
		//b.addi(gradient.getbGradient());
		this.biasTerms.plus(gradient.getbGradient());

	}


	public LogisticRegressionGradient getGradient(double lr) {
		
		//Matrix p_y_given_x = sigmoid(input.mmul(W).addRowVector(b));
		Matrix p_y_given_x = MatrixUtils.sigmoid( MatrixUtils.addRowVector( input.times( this.connectionWeights ), this.biasTerms.viewRow(0) ) );
		
		//Matrix dy = labels.sub(p_y_given_x);
		Matrix dy = labels.minus(p_y_given_x);
		
		if (useRegularization) {
			dy.divide( this.input.numRows() );
		}
		
		//Matrix wGradient = input.transpose().mmul(dy).mul(lr);
		Matrix wGradient = input.transpose().times( dy ).times( lr );
		
		Matrix bGradient = dy;
		return new LogisticRegressionGradient( wGradient, bGradient );
		
		
	}



	/**
	 * Classify input
	 * @param x the input (can either be a matrix or vector)
	 * If it's a matrix, each row is considered an example
	 * and associated rows are classified accordingly.
	 * Each row will be the likelihood of a label given that example
	 * @return a probability distribution for each row
	 */
	public Matrix predict(Matrix x) {
		
		//return softmax(x.mmul(W).addRowVector(b));
		return MatrixUtils.softmax( MatrixUtils.addRowVector( x.times( this.connectionWeights ), this.biasTerms.viewRow(0) ) );
		
	}	
	
	@Override
	protected LogisticRegression clone()  {
		LogisticRegression reg = new LogisticRegression();
		reg.biasTerms = biasTerms.clone();
		reg.connectionWeights = connectionWeights.clone();
		reg.l2 = this.l2;
		reg.labels = this.labels.clone();
		reg.nIn = this.nIn;
		reg.nOut = this.nOut;
		reg.useRegularization = this.useRegularization;
		reg.input = this.input.clone();
		return reg;
	}	
	
	/**
	 * Serializes this to the output stream.
	 * @param os the output stream to write to
	 */
	public void write(OutputStream os) {
		try {

		    DataOutput d = new DataOutputStream(os);
		    
		    d.writeInt( this.nIn );
		    d.writeInt( this.nOut );
		    
		    d.writeDouble( this.l2 );
			
		    MatrixWritable.writeMatrix(d, this.input );
		    MatrixWritable.writeMatrix(d, this.labels );			
		    MatrixWritable.writeMatrix(d, this.connectionWeights );
		    MatrixWritable.writeMatrix(d, this.biasTerms );
		    
		    d.writeBoolean( this.useRegularization );
		    

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
			
			this.nIn = di.readInt();
			this.nOut = di.readInt();
			
			this.l2 = di.readDouble();

			this.input = MatrixWritable.readMatrix( di );
			this.labels = MatrixWritable.readMatrix( di );
			this.connectionWeights = MatrixWritable.readMatrix( di );
			this.biasTerms = MatrixWritable.readMatrix( di );
			
			this.useRegularization = di.readBoolean();
			
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}	


/*
	public static class Builder {
		private Matrix W;
		private LogisticRegression ret;
		private Matrix b;
		private int nIn;
		private int nOut;
		private Matrix input;


		public Builder withWeights(Matrix W) {
			this.W = W;
			return this;
		}

		public Builder withBias(Matrix b) {
			this.b = b;
			return this;
		}

		public Builder numberOfInputs(int nIn) {
			this.nIn = nIn;
			return this;
		}

		public Builder numberOfOutputs(int nOut) {
			this.nOut = nOut;
			return this;
		}

		public LogisticRegression build() {
			ret = new LogisticRegression(input, nIn, nOut);
			ret.W = W;
			ret.b = b;
			return ret;
		}

	}
	*/
	
}
