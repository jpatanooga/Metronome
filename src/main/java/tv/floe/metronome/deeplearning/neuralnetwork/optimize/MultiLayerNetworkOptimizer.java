package tv.floe.metronome.deeplearning.neuralnetwork.optimize;

import cc.mallet.optimize.InvalidOptimizableException;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.OptimizationException;
import cc.mallet.optimize.Optimizer;

import java.io.Serializable;

import org.apache.mahout.math.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import tv.floe.metronome.deeplearning.neuralnetwork.core.BaseMultiLayerNeuralNetworkVectorized;
import tv.floe.metronome.deeplearning.neuralnetwork.optimize.util.CustomConjugateGradient;
import tv.floe.metronome.math.MatrixUtils;


public class MultiLayerNetworkOptimizer implements Optimizable.ByGradientValue,Serializable {

	private static final long serialVersionUID = 7607015610273177997L;

	protected BaseMultiLayerNeuralNetworkVectorized network;

	private static Logger log = LoggerFactory.getLogger(MultiLayerNetworkOptimizer.class);
	private double learningRate;
	private transient Optimizer optimizer;
	
	public MultiLayerNetworkOptimizer(BaseMultiLayerNeuralNetworkVectorized net, double lr) {
		
		this.network = net;
		this.learningRate = lr;
		
	}
/*	
	public void optimize(Matrix labels, double learningRate, int epochs) {
		
		MatrixUtils.ensureValidOutcomeMatrix(labels);
		//sample from the final layer in the network and train on the result
		Matrix layerInput = network.hiddenLayers[network.hiddenLayers.length - 1].sampleHiddenGivenLastVisible();
		
		network.outputLogisticLayer.inputTrainingData = layerInput;
		network.outputLogisticLayer.outputTrainingLabels = labels;

		
		Matrix w = network.outputLogisticLayer.connectionWeights.clone();
		Matrix b = network.outputLogisticLayer.biasTerms.clone();
		
		Double currLoss = null;
		Integer numTimesOver = null;

		for(int i = 0; i < epochs; i++) {

			network.outputLogisticLayer.train( layerInput, labels, learningRate );
			learningRate *= network.learningRateUpdate;
			
			if (currLoss == null) {
				
				currLoss = network.negativeLogLikelihood();
				
			} else {
				
				Double loss = network.negativeLogLikelihood();
				
				if (loss > currLoss) {
					
					if (numTimesOver == null) {
						
						numTimesOver = 1;
						
					} else {
						
						numTimesOver++;
						
					}
					
					if (numTimesOver >= 5) {
						
						log.info("Reverting weights and exiting...");
						network.outputLogisticLayer.connectionWeights = w.clone();
						network.outputLogisticLayer.biasTerms = b.clone();
						break;
						
					}
					
				} else if (loss < currLoss) {
					
					w = network.outputLogisticLayer.connectionWeights.clone();
					b = network.outputLogisticLayer.biasTerms.clone();
					currLoss = loss;
					
				}



			}

			log.info("Negative log likelihood on epoch " + i + " " + network.negativeLogLikelihood());
		}
		
		double curr = network.negativeLogLikelihood();
		
		if (curr > currLoss) {
			
			network.outputLogisticLayer.connectionWeights = w.clone();
			network.outputLogisticLayer.biasTerms = b.clone();
			log.info("Reverting to last known good state; converged after global minimum");
			
		}
		
		

		
	}	
	*/
	
	public void optimize(Matrix labels, double learningRate, int epochs) {
		
		MatrixUtils.ensureValidOutcomeMatrix(labels);
		//sample from the final layer in the network and train on the result
		Matrix layerInput = network.hiddenLayers[network.hiddenLayers.length - 1].sampleHiddenGivenLastVisible();
		
		network.outputLogisticLayer.inputTrainingData = layerInput;
		network.outputLogisticLayer.outputTrainingLabels = labels;

		
		Matrix w = network.outputLogisticLayer.connectionWeights.clone();
		Matrix b = network.outputLogisticLayer.biasTerms.clone();
		
		LogisticRegressionOptimizer opt = new LogisticRegressionOptimizer( network.outputLogisticLayer, learningRate );
		CustomConjugateGradient g = new CustomConjugateGradient(opt);
		g.optimize();
		
		network.backProp(lr, epochs);
		

	}
	
	
	@Override
	public int getNumParameters() {
		// 		return network.logLayer.W.length + network.logLayer.b.length;

		int connectionSize = network.outputLogisticLayer.connectionWeights.numRows() * network.outputLogisticLayer.connectionWeights.numCols(); 
		int biasSize = network.outputLogisticLayer.biasTerms.numRows() * network.outputLogisticLayer.biasTerms.numCols();
		return connectionSize + biasSize;
	}

	@Override
	public double getParameter(int index) {
		

		if (index > MatrixUtils.length( network.outputLogisticLayer.connectionWeights ) ) {
			
			int i = index - MatrixUtils.length( network.outputLogisticLayer.biasTerms );
			return MatrixUtils.getElement( network.outputLogisticLayer.biasTerms, i );
			
		} else {
			
			return MatrixUtils.getElement( network.outputLogisticLayer.connectionWeights, index );
			
		}
		
		
	}

	@Override
	public void getParameters(double[] buffer) {

		int idx = 0;
		
		for(int i = 0; i < MatrixUtils.length( network.outputLogisticLayer.connectionWeights ); i++) {
			buffer[idx++] = MatrixUtils.getElement( network.outputLogisticLayer.connectionWeights, i );
		}
		for(int i = 0; i < MatrixUtils.length( network.outputLogisticLayer.biasTerms ); i++) {
			buffer[idx++] = MatrixUtils.getElement( network.outputLogisticLayer.biasTerms, i);
		}		
		
	}

	@Override
	public void setParameter(int index, double value) {
		
		if (index >= MatrixUtils.length( network.outputLogisticLayer.connectionWeights ) ) {
			
			int i = index - MatrixUtils.length( network.outputLogisticLayer.biasTerms );
			MatrixUtils.setElement( network.outputLogisticLayer.biasTerms, i, value );
			
		} else {
			
			MatrixUtils.setElement( network.outputLogisticLayer.connectionWeights, index, value );
			
		}
		
	}

	@Override
	public void setParameters(double[] params) {
		
		int idx = 0;
		
		for (int i = 0; i < MatrixUtils.length( network.outputLogisticLayer.connectionWeights ); i++) {
			MatrixUtils.setElement( network.outputLogisticLayer.connectionWeights, i,params[idx++] );
		}

		for (int i = 0; i < MatrixUtils.length( network.outputLogisticLayer.biasTerms ); i++) {
			MatrixUtils.setElement( network.outputLogisticLayer.biasTerms, i, params[idx++] );
		}		
		
		
	}

	@Override
	public double getValue() {

		return network.negativeLogLikelihood();
		
	}

	/**
	 * TODO: Explain what is going on here and how this works!
	 * 
	 * 
	 * 
	 * 
	 * 
	 * 
	 * 
	 */
	@Override
	public void getValueGradient(double[] buffer) {
/*		
		

		
		DoubleMatrix weightGradient = network.logLayer.input.transpose().mmul(dy).mul(lr);
		DoubleMatrix biasGradient =  dy.columnMeans().mul(lr);
		for(int i = 0; i < weightGradient.length; i++)
			buffer[idx++] = weightGradient.get(i);
		for(int i = 0; i < biasGradient.length; i++)
			buffer[idx++] = biasGradient.get(i);
			*/
		
		// calc p_y_given_x
//				DoubleMatrix p_y_given_x = softmax(network.logLayer.input.mmul(network.logLayer.W).addRowVector(network.logLayer.b));

		Matrix p_y_given_x = MatrixUtils.softmax( MatrixUtils.addRowVector( network.outputLogisticLayer.inputTrainingData.times(network.outputLogisticLayer.connectionWeights ), network.outputLogisticLayer.biasTerms.viewRow(0) ) );
		
		// calc dy
		
		// DoubleMatrix dy = network.logLayer.labels.sub(p_y_given_x);
		
		Matrix dy = network.outputLogisticLayer.outputTrainingLabels.minus(p_y_given_x);

		// calc weight gradient
		
		int idx = 0;
		
		Matrix weightGradient = network.outputLogisticLayer.inputTrainingData.transpose().times(dy).times(learningRate);
		
		
	
		// calc bias gradient
		
		Matrix biasGradient = MatrixUtils.columnMeans(dy).times(learningRate);
		
/*
 * 
		for(int i = 0; i < weightGradient.length; i++)
			buffer[idx++] = weightGradient.get(i);
		for(int i = 0; i < biasGradient.length; i++)
			buffer[idx++] = biasGradient.get(i);

 * 				
 */
		
		for ( int i = 0; i < MatrixUtils.length( weightGradient ); i++ ) {
			
			buffer[ idx++ ] = MatrixUtils.getElement( weightGradient, i );
			
		}
		
		for ( int i = 0; i < MatrixUtils.length( biasGradient ); i++ ) {
			
			buffer[ idx++ ] = MatrixUtils.getElement( biasGradient, i );
			
		}
		
		
	}


}
