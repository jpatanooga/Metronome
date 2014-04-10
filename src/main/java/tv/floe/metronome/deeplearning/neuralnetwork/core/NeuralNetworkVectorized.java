package tv.floe.metronome.deeplearning.neuralnetwork.core;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.neuralnetwork.core.learning.AdagradLearningRate;
import tv.floe.metronome.deeplearning.neuralnetwork.gradient.NeuralNetworkGradient;
import tv.floe.metronome.types.Pair;





/**
 * Neural Network implementation (vectorization)
 * - designed to do work on all of the input samples at the same time via matrix ops
 * 
 * This namespace is designed differently from the mainline NN OOP-implementation
 * 
 * 
 * @author josh
 *
 */
public interface NeuralNetworkVectorized {

	
	public static enum OptimizationAlgorithm {
		GRADIENT_DESCENT,CONJUGATE_GRADIENT
	}

	public static enum LossFunction {
		SQUARED_LOSS,RECONSTRUCTION_CROSSENTROPY,NEGATIVELOGLIKELIHOOD
	}
	
	public boolean normalizeByInputRows();
	
	
	
	public abstract int getnVisible();

	public abstract void setnVisible(int nVisible);

	public abstract int getnHidden();

	public abstract void setnHidden(int nHidden);

	public abstract Matrix getConnectionWeights();

	public abstract void setConnectionWeights(Matrix weights);

	public abstract Matrix getHiddenBias();

	public abstract void sethBias(Matrix hiddenBias);

	public abstract Matrix getVisibleBias();

	public abstract void setVisibleBias(Matrix visibleBias);

	public abstract RandomGenerator getRng();

	public abstract void setRng(RandomGenerator rng);

	public abstract Matrix getInput();

	public abstract void setInput(Matrix input);
	
	public double getSparsity();
	public abstract void setSparsity(double sparsity);
	
	public NeuralNetworkGradient getGradient(Object[] params);
	
	
	public double getL2();
	public void setL2(double l2);
	
	public double getMomentum();
	public void setMomentum(double momentum);
	
	public void setRenderEpochs(int renderEpochs);
	public int getRenderEpochs();

	public NeuralNetworkVectorized transpose();
	public  NeuralNetworkVectorized clone();

	public double fanIn();
	public void setFanIn(double fanIn);
	
	public double squaredLoss();
	
	
	public double l2RegularizedCoefficient();
	
	public double getReConstructionCrossEntropy();	
	public void trainTillConvergence(Matrix input, double lr, Object[] params);
	public void train(Matrix input,double lr,Object[] params);
	
	/**
	 * Performs a network merge in the form of
	 * a += b - a / n
	 * where a is a matrix here
	 * b is a matrix on the incoming network
	 * and n is the batch size
	 * @param network the network to merge with
	 * @param batchSize the batch size (number of training examples)
	 * to average by
	 */
	public void merge(NeuralNetworkVectorized network,int batchSize);	
	
	public void clearWeights();
	
	public AdagradLearningRate getAdaGrad();
	public void setAdaGrad(AdagradLearningRate adaGrad);
	
    public boolean isUseRegularization();

	public AdagradLearningRate gethBiasAdaGrad();
	public void setHbiasAdaGrad(AdagradLearningRate adaGrad);
	
	
	public AdagradLearningRate getVBiasAdaGrad();
	public void setVBiasAdaGrad(AdagradLearningRate adaGrad);
	
	
	/**
	 * Sample hidden mean and sample
	 * given visible
	 * @param v the  the visible input
	 * @return a pair with mean, sample
	 */
	public Pair<Matrix,Matrix> sampleHiddenGivenVisible(Matrix v);


    public boolean isUseAdaGrad();
	
	public void setDropOut(double dropOut);
	public double dropOut();
	
	/**
	 * Sample visible mean and sample
	 * given hidden
	 * @param h the  the hidden input
	 * @return a pair with mean, sample
	 */
	public Pair<Matrix,Matrix> sampleVisibleGivenHidden(Matrix h);    
	
	
}
