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
		
		network.logisticRegressionLayer.input = layerInput;
		network.logisticRegressionLayer.labels = labels;

		
//		Matrix w = network.logisticRegressionLayer.connectionWeights.clone();
//		Matrix b = network.logisticRegressionLayer.biasTerms.clone();
		
		LogisticRegressionOptimizer opt = new LogisticRegressionOptimizer( network.logisticRegressionLayer, learningRate );
		CustomConjugateGradient g = new CustomConjugateGradient(opt);
		g.optimize();
		System.out.println("using LogisticRegressionOptimizer and CustomConjugateGradient !!!");
		
		network.backProp(learningRate, epochs);
		

	}
	
	
	@Override
	public int getNumParameters() {
		// 		return network.logLayer.W.length + network.logLayer.b.length;

		int connectionSize = network.logisticRegressionLayer.connectionWeights.numRows() * network.logisticRegressionLayer.connectionWeights.numCols(); 
		int biasSize = network.logisticRegressionLayer.biasTerms.numRows() * network.logisticRegressionLayer.biasTerms.numCols();
		return connectionSize + biasSize;
	}

	@Override
	public double getParameter(int index) {
		

		if (index > MatrixUtils.length( network.logisticRegressionLayer.connectionWeights ) ) {
			
			int i = index - MatrixUtils.length( network.logisticRegressionLayer.biasTerms );
			return MatrixUtils.getElement( network.logisticRegressionLayer.biasTerms, i );
			
		} else {
			
			return MatrixUtils.getElement( network.logisticRegressionLayer.connectionWeights, index );
			
		}
		
		
	}

	@Override
	public void getParameters(double[] buffer) {

		int idx = 0;
		
		for(int i = 0; i < MatrixUtils.length( network.logisticRegressionLayer.connectionWeights ); i++) {
			buffer[idx++] = MatrixUtils.getElement( network.logisticRegressionLayer.connectionWeights, i );
		}
		for(int i = 0; i < MatrixUtils.length( network.logisticRegressionLayer.biasTerms ); i++) {
			buffer[idx++] = MatrixUtils.getElement( network.logisticRegressionLayer.biasTerms, i);
		}		
		
	}

	@Override
	public void setParameter(int index, double value) {
		
		if (index >= MatrixUtils.length( network.logisticRegressionLayer.connectionWeights ) ) {
			
			int i = index - MatrixUtils.length( network.logisticRegressionLayer.biasTerms );
			MatrixUtils.setElement( network.logisticRegressionLayer.biasTerms, i, value );
			
		} else {
			
			MatrixUtils.setElement( network.logisticRegressionLayer.connectionWeights, index, value );
			
		}
		
	}

	@Override
	public void setParameters(double[] params) {
		
		int idx = 0;
		
		for (int i = 0; i < MatrixUtils.length( network.logisticRegressionLayer.connectionWeights ); i++) {
			MatrixUtils.setElement( network.logisticRegressionLayer.connectionWeights, i,params[idx++] );
		}

		for (int i = 0; i < MatrixUtils.length( network.logisticRegressionLayer.biasTerms ); i++) {
			MatrixUtils.setElement( network.logisticRegressionLayer.biasTerms, i, params[idx++] );
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

		Matrix p_y_given_x = MatrixUtils.softmax( MatrixUtils.addRowVector( network.logisticRegressionLayer.input.times(network.logisticRegressionLayer.connectionWeights ), network.logisticRegressionLayer.biasTerms.viewRow(0) ) );
		
		// calc dy
		
		// DoubleMatrix dy = network.logLayer.labels.sub(p_y_given_x);
		
		Matrix dy = network.logisticRegressionLayer.labels.minus(p_y_given_x);

		// calc weight gradient
		
		int idx = 0;
		
		Matrix weightGradient = network.logisticRegressionLayer.input.transpose().times(dy).times(learningRate);
		
		
	
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
