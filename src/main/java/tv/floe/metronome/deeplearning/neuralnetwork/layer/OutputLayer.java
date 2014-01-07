package tv.floe.metronome.deeplearning.neuralnetwork.layer;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

import tv.floe.metronome.math.MatrixUtils;


public class OutputLayer {


	private static final long serialVersionUID = -7065564817460914364L;
	public int numInputNeurons;
	public int numOutputNeurons;
	
	public Matrix inputTrainingData = null;
	public Matrix outputTrainingLabels = null;
	
	public Matrix connectionWeights;
	public Matrix biasTerms;



	public OutputLayer(Matrix input,Matrix labels, int nIn, int nOut) {
		this.inputTrainingData = input;
		this.outputTrainingLabels = labels;
		this.numInputNeurons = nIn;
		this.numOutputNeurons = nOut;
		//connectionWeights = Matrix.zeros( nIn, nOut );
		this.connectionWeights = new DenseMatrix( nIn, nOut );
		this.connectionWeights.assign(0.0);
		
		this.biasTerms = new DenseMatrix(nOut, 1); //Matrix.zeros(nOut);
		this.biasTerms.assign(0.0);
	}

	public OutputLayer(Matrix input, int nIn, int nOut) {
		this( input, null, nIn, nOut );
	}


	public void train(double learningRate) {
		train( inputTrainingData, outputTrainingLabels, learningRate );
	}


	public void train( Matrix x, double learningRate ) {
		train( x, outputTrainingLabels, learningRate );

	}
	
/*	public void merge(HiddenLayer layer,int batchSize) {
		connectionWeights.addi(layer.connectionWeights.subi(connectionWeights).div(batchSize));
		biasTerms.addi(layer.biasTerms.subi(biasTerms).div(batchSize));
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
/*		
		return - labels.mul(log(sigAct)).add(
				oneMinus(labels).mul(
						log(oneMinus(sigAct))
				))
				.columnSums().mean();
				*/
		
		Matrix eleMul = MatrixUtils.elementWiseMultiplication( this.outputTrainingLabels, MatrixUtils.log(sigAct) );

//		oneMinus(labels).mul(
//				log(oneMinus(sigAct))
		Matrix oneMinusLabels = MatrixUtils.oneMinus( this.outputTrainingLabels );
		Matrix logOneMinusSigAct = MatrixUtils.log( MatrixUtils.oneMinus(sigAct) );
		Matrix labelsMulSigAct = MatrixUtils.elementWiseMultiplication(oneMinusLabels, logOneMinusSigAct);

		Matrix sum = eleMul.plus(labelsMulSigAct);
		//Matrix columnSum = MatrixUtils.co
		
		// TODO: finish column sums
		
		// TODO: finish mean
		
		// TODO: negative output
		
		
	}

	/**
	 * Train on the given inputs and labels.
	 * This connectionWeightsill assign the passed in values
	 * as fields to this logistic function for 
	 * caching.
	 * @param x the inputs to train on
	 * @param y the labels to train on
	 * @param lr the learning rate
	 */
	public void train(Matrix x, Matrix y, double lr) {
		
		//ensureValidOutcomeMatrix(y);

		this.inputTrainingData = x;
		this.labels = y;

		if(x.rows != y.rows)
			throw new IllegalArgumentException("Can't train on the 2 given inputs and labels");

		Matrix p_y_given_x = softmax(x.mmul(connectionWeights).addRoconnectionWeightsVector(biasTerms));
		
		Matrix dy = y.sub(p_y_given_x);

		connectionWeights = connectionWeights.add(x.transpose().mmul(dy).mul(lr));
		biasTerms = biasTerms.add(dy.columnMeans().mul(lr));

	}





	/**
	 * Classify input
	 * @param x the input (can either be a matrix or vector)
	 * If it's a matrix, each roconnectionWeights is considered an example
	 * and associated roconnectionWeightss are classified accordingly.
	 * Each roconnectionWeights connectionWeightsill be the likelihood of a label given that example
	 * @return a probability distribution for each roconnectionWeights
	 */
	public Matrix predict(Matrix x) {
		return softmax(x.mmul(connectionWeights).addRoconnectionWeightsVector(biasTerms));
	}	

}
