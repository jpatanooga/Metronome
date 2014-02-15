package tv.floe.metronome.deeplearning.neuralnetwork.layer;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;

import tv.floe.metronome.math.MatrixUtils;

/**
 * 
 * DEPRECATED - Not used
 * 
 * @author josh
 *
 */
@Deprecated
public class LogisticRegressionLayer {

	//private static final long serialVersionUID = -7065564817460914364L;
	public int numInputNeurons;
	public int numOutputNeurons;
	
	public Matrix inputTrainingData = null;
	public Matrix outputTrainingLabels = null;
	
	public Matrix connectionWeights;
	public Matrix biasTerms;



	public LogisticRegressionLayer(Matrix input,Matrix labels, int nIn, int nOut) {
		this.inputTrainingData = input;
		this.outputTrainingLabels = labels;
		this.numInputNeurons = nIn;
		this.numOutputNeurons = nOut;
		this.connectionWeights = new DenseMatrix( nIn, nOut );
		this.connectionWeights.assign(0.0);
		
		this.biasTerms = new DenseMatrix(1, nOut); //Matrix.zeros(nOut);
		this.biasTerms.assign(0.0);
	}

	public LogisticRegressionLayer(Matrix input, int nIn, int nOut) {
		this(input,null,nIn,nOut);
	}


	public void train(double learningRate) {
		//train(input,labels,lr);
		train( inputTrainingData, outputTrainingLabels, learningRate );

	}


	public void train( Matrix x, double learningRate ) {

		train( x, outputTrainingLabels, learningRate );

	}
/*	
	public void merge(LogisticRegression l,int batchSize) {
		W.addi(l.W.subi(W).div(batchSize));
		b.addi(l.b.subi(b).div(batchSize));
	}
*/
	/**
	 * Objective function:  minimize negative log likelihood
	 * @return the negative log likelihood of the model
	 */
	public double negativeLogLikelihood() {
				
		Matrix mult = this.inputTrainingData.times(connectionWeights);
		Matrix multPlusBias = MatrixUtils.addRowVector(mult, this.biasTerms.viewRow(0));
		Matrix sigAct = MatrixUtils.softmax(multPlusBias); 
		
		Matrix eleMul = MatrixUtils.elementWiseMultiplication( this.outputTrainingLabels, MatrixUtils.log(sigAct) );

		Matrix oneMinusLabels = MatrixUtils.oneMinus( this.outputTrainingLabels );
		Matrix logOneMinusSigAct = MatrixUtils.log( MatrixUtils.oneMinus(sigAct) );
		Matrix labelsMulSigAct = MatrixUtils.elementWiseMultiplication(oneMinusLabels, logOneMinusSigAct);

		Matrix sum = eleMul.plus(labelsMulSigAct);
		
		Matrix colSumsMatrix = MatrixUtils.columnSums(sum);
		
		double matrixMean = MatrixUtils.mean(colSumsMatrix);
		
		return -matrixMean;
		
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
	/**
	 * Train on the given inputs and labels.
	 * 
	 * This will assign the passed in values as fields to this logistic function for 
	 * caching.
	 * 
	 * @param input the inputs to train on
	 * @param labels the labels to train on
	 * @param lr the learning rate
	 */
	public void train(Matrix input, Matrix labels, double lr) {
		
		MatrixUtils.ensureValidOutcomeMatrix(labels);

		this.inputTrainingData = input;
		this.outputTrainingLabels = labels;

		if (input.numRows() != labels.numRows()) {
			throw new IllegalArgumentException("Can't train on the 2 given inputs and labels");
		}

		//Matrix p_y_given_x = softmax(input.mmul(connectionWeights).addRoconnectionWeightsVector(biasTerms));
		Matrix p_LabelsGivenInput = input.times(this.connectionWeights);
		p_LabelsGivenInput = MatrixUtils.softmax(MatrixUtils.addRowVector(p_LabelsGivenInput, this.biasTerms.viewRow(0)));
		
		//Matrix dy = y.sub(p_y_given_x);
		Matrix dy = labels.minus(p_LabelsGivenInput);

		//connectionWeights = connectionWeights.add(x.transpose().mmul(dy).mul(lr));		
		Matrix baseConnectionUpdate = input.transpose().times(dy);
		this.connectionWeights = this.connectionWeights.plus( baseConnectionUpdate.times(lr) );
		
		//biasTerms = biasTerms.add(dy.columnMeans().mul(lr));
		this.biasTerms = this.biasTerms.plus( MatrixUtils.columnMeans(dy).times(lr) );

	}





	/**
	 * Classify input
	 * @param input (can either be a matrix or vector)
	 * If it's a matrix, each row is considered an example
	 * and associated rows are classified accordingly.
	 * Each row will be the likelihood of a label given that example
	 * @return a probability distribution for each row
	 */
/*	public Matrix predict(Matrix x) {
		return softmax(x.mmul(W).addRowVector(b));
	}	
	*/
	public Matrix predict(Matrix input) {
		
		Matrix prediction = input.times(this.connectionWeights);
		prediction = MatrixUtils.softmax(MatrixUtils.addRowVector(prediction, this.biasTerms.viewRow(0)));
		
		return prediction;
		
	}	
	
	
	
	/**
	 * Serializes this to the output stream.
	 * @param os the output stream to write to
	 */
	public void write(OutputStream os) {
		try {

		    DataOutput d = new DataOutputStream(os);
		    
		    d.writeInt( this.numInputNeurons );
		    d.writeInt( this.numOutputNeurons );
			
		    MatrixWritable.writeMatrix(d, this.inputTrainingData );
		    MatrixWritable.writeMatrix(d, this.outputTrainingLabels );
			
		    MatrixWritable.writeMatrix(d, this.connectionWeights );
		    MatrixWritable.writeMatrix(d, this.biasTerms );
		    
		    

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
			
			this.numInputNeurons = di.readInt();
			this.numOutputNeurons = di.readInt();

			this.inputTrainingData = MatrixWritable.readMatrix( di );
			this.outputTrainingLabels = MatrixWritable.readMatrix( di );
			
			this.connectionWeights = MatrixWritable.readMatrix( di );
			this.biasTerms = MatrixWritable.readMatrix( di );
			
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}
		
	
	
	
	
}
