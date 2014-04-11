package tv.floe.metronome.deeplearning.neuralnetwork.core;


import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.neuralnetwork.core.learning.AdagradLearningRate;
import tv.floe.metronome.deeplearning.neuralnetwork.gradient.NeuralNetworkGradient;
import tv.floe.metronome.math.MathUtils;
import tv.floe.metronome.math.MatrixUtils;

/**
 * Based on the NN design by Adam Gibson
 * 
 * 
 * 
 * Connections are based on a concept of a matrix  of size ( visibleNeurons x hiddenNeurons )
 * - where each row represents the connections for a specific visible neuron (n-th) to all other hidden neurons (m-th) 
 * 
 * Hidden and Visible Neurons
 * - these are 1 x NeuronCount dim Matrices (vectors, really) that hold the states of each visible or hidden neuron based on activations
 * 
 * @author josh
 *
 */
public abstract class BaseNeuralNetworkVectorized implements NeuralNetworkVectorized {

	//public int inputNeuronCount;
	
	public int numberVisibleNeurons;
	public int numberHiddenNeurons;
	
	public Matrix hiddenBiasNeurons;
	public Matrix visibleBiasNeurons;
	public Matrix connectionWeights;
	public Matrix trainingDataset = null;	
		
	public RandomGenerator randNumGenerator;
	

	public double sparsity = 0.01;
	/* momentum for learning */
	public double momentum = 0.1;
	/* L2 Regularization constant */
	public double l2 = 0.1;

	public int renderWeightsEveryNumEpochs = -1;
	
	public double fanIn = -1;

	protected boolean useRegularization = false;
    protected boolean useAdaGrad = false;
	
	private AdagradLearningRate wAdagrad = null; 
	private AdagradLearningRate hBiasAdaGrad = null;
	private AdagradLearningRate vBiasAdaGrad = null;

    protected boolean firstTimeThrough = false;
    //normalize by input rows or not
    protected boolean normalizeByInputRows = false;
    //use only when binary hidden layers are active
    protected boolean applySparsity = true;

    protected double dropOut = 0;
    protected Matrix doMask;

    protected OptimizationAlgorithm optimizationAlgo;
    protected LossFunction lossFunction;

	
	
	// default CTOR
	public BaseNeuralNetworkVectorized() {
		
	}
	
	public BaseNeuralNetworkVectorized(int nVisible, int nHidden, Matrix weights, Matrix hBias, Matrix vBias, RandomGenerator rng) {
		this.numberVisibleNeurons = nVisible;
		this.numberHiddenNeurons = nHidden;

		if (rng == null)	{
			this.randNumGenerator = new MersenneTwister(1234);
		} else { 
			this.randNumGenerator = rng;
		}

		if (weights == null) {
			
			double a = 1.0 / (double) nVisible;
			/*
			 * Initialize based on the number of visible units..
			 * The lower bound is called the fan in
			 * The outer bound is called the fan out.
			 * 
			 * Below's advice works for Denoising AutoEncoders and other 
			 * neural networks you will use due to the same baseline guiding principles for
			 * both RBMs and Denoising Autoencoders.
			 * 
			 * Hinton's Guide to practical RBMs:
			 * The weights are typically initialized to small random values chosen from a zero-mean Gaussian with
			 * a standard deviation of about 0.01. Using larger random values can speed the initial learning, but
			 * it may lead to a slightly worse final model. Care should be taken to ensure that the initial weight
			 * values do not allow typical visible vectors to drive the hidden unit probabilities very close to 1 or 0
			 * as this significantly slows the learning.
			 */
			NormalDistribution u = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);

			//this.connectionWeights = Matrix.zeros(nVisible,nHidden);
			this.connectionWeights = new DenseMatrix( nVisible, nHidden );
			this.connectionWeights.assign(0);

			for(int i = 0; i < this.connectionWeights.numRows(); i++) {
				for(int j = 0; j < this.connectionWeights.numCols(); j++) {
					this.connectionWeights.set(i,j,u.sample());
				}
			}


		} else {	
			this.connectionWeights = weights;
		}
		

		this.wAdagrad = new AdagradLearningRate( this.connectionWeights.numRows(), this.connectionWeights.numCols() );


		if (hBias == null) { 
			// TODO: recheck if this column vector is correctly oriented
			this.hiddenBiasNeurons = new DenseMatrix(1, nHidden); //Matrix.zeros(nHidden);
		//} else if(hBias.numRows() != nHidden) {
			//throw new IllegalArgumentException("Hidden bias must have a length of " + nHidden + " length was " + hBias.numRows());
		} else {
			this.hiddenBiasNeurons = hBias;
		}
		
		//         this.hBiasAdaGrad = new AdaGrad(hBias.rows,hBias.columns);

		this.hBiasAdaGrad = new AdagradLearningRate( this.hiddenBiasNeurons.numRows(), this.hiddenBiasNeurons.numCols() );

		if (vBias == null) { 
			this.visibleBiasNeurons = new DenseMatrix(1, nVisible); //Matrix.zeros(nVisible);
			this.visibleBiasNeurons.assign(0);

		} else if(vBias.numRows() != nVisible) { 
			throw new IllegalArgumentException("Visible bias must have a length of " + nVisible + " but length was " + vBias.numRows());

		} else { 
			this.visibleBiasNeurons = vBias;
		}
		
		// this.vBiasAdaGrad = new AdaGrad(vBias.rows,vBias.columns);
		
		this.vBiasAdaGrad = new AdagradLearningRate( this.visibleBiasNeurons.numRows(), this.visibleBiasNeurons.numCols() );
		
	}	
	
	public BaseNeuralNetworkVectorized(Matrix input, int nVisible, int nHidden, Matrix weights, Matrix hBias, Matrix vBias, RandomGenerator rng) {

		this(nVisible, nHidden, weights, hBias, vBias, rng);
		this.trainingDataset = input;
	}
	
	
	@Override
	public int getnVisible() {

		return this.numberVisibleNeurons;
		
	}

	@Override
	public void setnVisible(int nVisible) {

		this.numberVisibleNeurons = nVisible;
		
	}

	@Override
	public int getnHidden() {
		
		return this.numberHiddenNeurons;
	}

	@Override
	public void setnHidden(int nHidden) {

		this.numberHiddenNeurons = nHidden;
		
	}

	@Override
	public Matrix getConnectionWeights() {

		return this.connectionWeights;
		
	}

	@Override
	public void setConnectionWeights(Matrix weights) {

		this.connectionWeights = weights;
		
	}

	@Override
	public Matrix getHiddenBias() {

		return this.hiddenBiasNeurons;
		
	}

	@Override
	public void sethBias(Matrix hiddenBias) {
		
		this.hiddenBiasNeurons = hiddenBias;
		
	}

	@Override
	public Matrix getVisibleBias() {

		return this.visibleBiasNeurons;
		
	}

	@Override
	public void setVisibleBias(Matrix visibleBias) {

		this.visibleBiasNeurons = visibleBias;
		
	}

	@Override
	public RandomGenerator getRng() {

		return this.randNumGenerator;
		
	}

	@Override
	public void setRng(RandomGenerator rng) {

		this.randNumGenerator = rng;
		
	}

	@Override
	public Matrix getInput() {
		// TODO Auto-generated method stub
		return this.trainingDataset;
	}

	@Override
	public void setInput(Matrix input) {

		this.trainingDataset = input;
		
	}
	
	@Override
	public NeuralNetworkVectorized transpose() {
		try {
			NeuralNetworkVectorized ret = getClass().newInstance();
			ret.sethBias( this.hiddenBiasNeurons.clone() );
			ret.setVisibleBias( this.visibleBiasNeurons.clone() );
			ret.setnHidden(getnVisible());
			ret.setnVisible(getnHidden());
			ret.setConnectionWeights( this.connectionWeights.transpose() );
			ret.setRng(getRng());

			// ret.setAdaGrad(wAdaGrad);
			ret.setAdaGrad( this.wAdagrad );
			
			return ret;
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 


	}

	@Override
	public NeuralNetworkVectorized clone() {
		try {
			NeuralNetworkVectorized ret = getClass().newInstance();
			ret.sethBias( this.hiddenBiasNeurons.clone() );
			ret.setVisibleBias( this.visibleBiasNeurons.clone() );
			ret.setnHidden(getnHidden());
			ret.setnVisible(getnVisible());
			//ret.setW(W.dup());
			ret.setConnectionWeights( this.connectionWeights.clone() );
			ret.setRng(getRng());
			
			ret.setAdaGrad( getAdaGrad().clone() );
			ret.setHbiasAdaGrad( this.gethBiasAdaGrad().clone() );
			ret.setVBiasAdaGrad( this.getVBiasAdaGrad().clone() );

            ret.setMomentum(momentum);
            //ret.setRenderEpochs(getRenderEpochs());
            ret.setSparsity(sparsity);
			
            ret.setLossFunction(lossFunction);
            ret.setOptimizationAlgorithm(optimizationAlgo);
            
			
			return ret;
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 


	}
	
	
    /**
     * Applies sparsity to the passed in hbias gradient
     * @param hBiasGradient the hbias gradient to apply to
     * @param learningRate the learning rate used
     */
    protected void applySparsity(Matrix hBiasGradient, double learningRate) {

        if (useAdaGrad) {
        	
        	//Matrix change = this.hBiasAdaGrad.getLearningRates(this.hiddenBiasNeurons).neg().mul(sparsity).mul(hBiasGradient.mul(sparsity));
        	Matrix change = MatrixUtils.elementWiseMultiplication( MatrixUtils.neg( this.hBiasAdaGrad.getLearningRates(this.hiddenBiasNeurons) ).times(sparsity), hBiasGradient.times(sparsity));
            // hBiasGradient.addi(change);
            MatrixUtils.addi( hBiasGradient, change );
            
        } else {
        	//Matrix change = hBiasGradient.mul(sparsity).mul(-learningRate * sparsity);
        	Matrix change = hBiasGradient.times(sparsity).times( -learningRate * sparsity );
            //hBiasGradient.addi(change);
        	MatrixUtils.addi( hBiasGradient, change );

        }
        
    }
	
    @Override
    public LossFunction getLossFunction() {
        return lossFunction;
    }
    @Override
    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }
    @Override
    public OptimizationAlgorithm getOptimizationAlgorithm() {
        return optimizationAlgo;
    }
    @Override
    public void setOptimizationAlgorithm(
            OptimizationAlgorithm optimizationAlgorithm) {
        this.optimizationAlgo = optimizationAlgorithm;
    }	

	@Override
	public double getSparsity() {
		return this.sparsity;
	}

	@Override
	public void setSparsity(double sparsity) {

		this.sparsity = sparsity;
	}

	@Override
	public double getL2() {
		return this.l2;
	}

	@Override
	public void setL2(double l2) {

		this.l2 = l2;
		
	}

	@Override
	public double getMomentum() {
		
		return this.momentum;
		
	}

	@Override
	public void setMomentum(double momentum) {

		this.momentum = momentum;
		
	}
	
    /**
     * Update the gradient according to the configuration such as adagrad, momentum, and sparsity
     * @param gradient the gradient to modify
     * @param learningRate the learning rate for the current iteratiaon
     */
    protected void updateGradientAccordingToParams(NeuralNetworkGradient gradient, double learningRate) {
        Matrix wGradient = gradient.getwGradient();
        Matrix hBiasGradient = gradient.gethBiasGradient();
        Matrix vBiasGradient = gradient.getvBiasGradient();
        Matrix wLearningRates = wAdagrad.getLearningRates(wGradient);
        
        if (useAdaGrad) {
            //wGradient.muli(wLearningRates);
        	wGradient = wGradient.times( wLearningRates );
            
        } else {
            //wGradient.muli(learningRate);
        	wGradient = wGradient.times( learningRate );
        }

        if (useAdaGrad) {
        	
            //hBiasGradient = hBiasGradient.mul(hBiasAdaGrad.getLearningRates(hBiasGradient)).add(hBiasGradient.mul(momentum));
        	hBiasGradient = hBiasGradient.times( hBiasAdaGrad.getLearningRates( hBiasGradient ) ).plus( hBiasGradient.times( momentum ) );
        
        } else {
            
        	//hBiasGradient = hBiasGradient.mul(learningRate).add(hBiasGradient.mul(momentum));
        	hBiasGradient = hBiasGradient.times( learningRate ).plus( hBiasGradient.times( momentum ) );
        	
        }


        if (useAdaGrad) {
        	
            //vBiasGradient = vBiasGradient.mul(vBiasAdaGrad.getLearningRates(vBiasGradient)).add(vBiasGradient.mul(momentum));
        	vBiasGradient = vBiasGradient.times( vBiasAdaGrad.getLearningRates( vBiasGradient ) ).plus( vBiasGradient.times( momentum ) );
        
        } else {
        
        	//vBiasGradient = vBiasGradient.mul(learningRate).add(vBiasGradient.mul(momentum));
        	vBiasGradient = vBiasGradient.times( learningRate ).plus( vBiasGradient.times( momentum ) );
        	
        }



        //only do this with binary hidden layers
        if (applySparsity) {
            applySparsity( hBiasGradient, learningRate );
        }

        if (momentum != 0) {
        	
            //Matrix change = wGradient.mul(momentum).add(wGradient.mul(1 - momentum));
        	Matrix change = wGradient.times( momentum ).plus(wGradient.times( 1 - momentum ) );
        	
            //wGradient.addi(change);
        	MatrixUtils.addi( wGradient, change );

        }

        if (useRegularization) {
        	
            if (l2 > 0) {
            	
            	//Matrix penalized = W.mul(l2);
            	Matrix penalized = this.connectionWeights.times( l2 );
                //wGradient.subi(penalized);
            	wGradient = wGradient.minus( penalized );

            }

        }


        if (normalizeByInputRows) {
        	
            // wGradient.divi(input.rows);
        	MatrixUtils.divi( wGradient, this.trainingDataset.numRows() );
            //vBiasGradient.divi(input.rows);
        	MatrixUtils.divi( vBiasGradient, this.trainingDataset.numRows() );
            //hBiasGradient.divi(input.rows);
        	MatrixUtils.divi( hBiasGradient, this.trainingDataset.numRows() );
            
        }

    }
    
    /**
     * Copies params from the passed in network
     * to this one
     * @param n the network to copy
     */
    public void update(BaseNeuralNetworkVectorized n) {
    	
        this.connectionWeights = n.connectionWeights;
        this.normalizeByInputRows = n.normalizeByInputRows;
        this.hiddenBiasNeurons = n.hiddenBiasNeurons;
        this.visibleBiasNeurons = n.visibleBiasNeurons;
        this.l2 = n.l2;
        this.useRegularization = n.useRegularization;
        this.momentum = n.momentum;
        this.numberHiddenNeurons = n.numberHiddenNeurons;
        this.numberVisibleNeurons = n.numberVisibleNeurons;
        this.randNumGenerator = n.randNumGenerator;
        this.sparsity = n.sparsity;
        this.wAdagrad = n.wAdagrad;
        this.hBiasAdaGrad = n.hBiasAdaGrad;
        this.vBiasAdaGrad = n.vBiasAdaGrad;
        this.optimizationAlgo = n.optimizationAlgo;
        this.lossFunction = n.lossFunction;
        
    }    
    
    @Override
    public double negativeLogLikelihood() {
    	
        Matrix z = this.reconstruct( this.trainingDataset );
        
        if (this.useRegularization) {
        	
            double reg = (2 / l2) * MatrixUtils.sum( MatrixUtils.pow(this.connectionWeights,2) );

//            double ret = - input.mul(log(z)).add(
//                    oneMinus(input).mul(log(oneMinus(z)))).
//                    columnSums().mean() + reg;
  
            Matrix tmpElementwiseMul = MatrixUtils.elementWiseMultiplication( MatrixUtils.oneMinus( this.trainingDataset ), MatrixUtils.log( MatrixUtils.oneMinus( z ) ) );
            double ret = - MatrixUtils.mean( MatrixUtils.columnSums( this.trainingDataset.times( MatrixUtils.log( z ) ).plus( tmpElementwiseMul ) ) ) + reg;
            
            
            if (this.normalizeByInputRows) {
            	
                ret /= this.trainingDataset.numRows();
                
            }
            
            return ret;
            
        }
/*
        double likelihood =  - input.mul(log(z)).add(
                oneMinus(input).mul(log(oneMinus(z)))).
                columnSums().mean();
*/
        
        // input.mul(log(z))
        Matrix tmplogZTimesInput = MatrixUtils.elementWiseMultiplication( this.trainingDataset, MatrixUtils.log( z ) );
        
        
        // log(oneMinus(z))
        Matrix tmpLogOneMinus = MatrixUtils.log( MatrixUtils.oneMinus( z ) );

       
        
        //  oneMinus(input).mul(log(oneMinus(z)))
        Matrix tmpOneMinus = MatrixUtils.oneMinus( this.trainingDataset );
        Matrix tmpElementwiseMul = MatrixUtils.elementWiseMultiplication( tmpOneMinus, tmpLogOneMinus );
        
        Matrix tmpPlusOneMinus = tmplogZTimesInput.plus( tmpElementwiseMul );

        
        double likelihood = - MatrixUtils.mean( MatrixUtils.columnSums( tmpPlusOneMinus ) );
        
        if (this.normalizeByInputRows) {
           
        	likelihood /= this.trainingDataset.numRows(); //input.rows;
        	
        }


        return likelihood;
    }    
	

    @Override
    public void setDropOut(double dropOut) {
        this.dropOut = dropOut;
    }
    
    @Override
    public double dropOut() {
        return dropOut;
    }	
	

/*	@Override
	public void trainTillConvergence(Matrix input, double lr, Object[] params) {
		// TODO Auto-generated method stub
		
	}	
	*/
	
	/**
	 * All neural networks are based on this idea of 
	 * minimizing reconstruction error.
	 * Both RBMs and Denoising AutoEncoders
	 * have a component for reconstructing, ala different implementations.
	 *  
	 * @param x the input to reconstruct
	 * @return the reconstructed input
	 */
	public abstract Matrix reconstruct(Matrix x);
	
	/**
	 * The loss function (cross entropy, reconstruction error,...)
	 * @return the loss function
	 */
	public abstract double lossFunction(Object[] params);

	public double lossFunction() {
		return lossFunction(null);
	}
	
	@Override
	public double squaredLoss() {
		Matrix reconstructed = reconstruct( this.trainingDataset );
		
		//double loss = MatrixFunctions.powi(reconstructed.sub(input), 2).sum() / input.rows;
		double loss = MatrixUtils.sum( MatrixUtils.pow( reconstructed.minus( this.trainingDataset ), 2 ) ) / this.trainingDataset.numRows();
		
		if(this.useRegularization) {
			//loss += 0.5 * l2 * MatrixFunctions.pow(W,2).sum();
			loss += 0.5 * l2 * MatrixUtils.sum( MatrixUtils.pow( this.connectionWeights, 2 ) );
		}
		
		return -loss;
	}
	
	
	/**
	 * Train one iteration of the network
	 * @param input the input to train on
	 * @param lr the learning rate to train at
	 * @param params the extra params (k, corruption level,...)
	 */
	public abstract void train(Matrix input, double learningRate, Object[] params);
		
    protected void applyDropOutIfNecessary(Matrix input) {
        if (dropOut > 0) {
         //   this.doMask = DoubleMatrix.rand(input.rows, this.nHidden).gt(dropOut);

        	this.doMask = MathUtils.rand( this.trainingDataset.numRows(), this.numberHiddenNeurons );
        	
        } else {

        //	this.doMask = DoubleMatrix.ones(input.rows,this.nHidden);
        	this.doMask = MatrixUtils.ones( this.trainingDataset.numRows(), this.numberHiddenNeurons );
        	
        }
    }
	
    @Override
    public Matrix hBiasMean() {
    	//Matrix hbiasMean = getInput().times( this.connectionWeights ).addRowVector( this.getHiddenBias() );
    	Matrix hbiasMean = MatrixUtils.addRowVector( getInput().times( this.connectionWeights ), this.getHiddenBias().viewRow(0) );
        return hbiasMean;
    }
	
	
	/**
	 * Reconstruction error.
	 * 
	 * Reconstruction entropy.
	 * This compares the similarity of two probability
	 * distributions, in this case that would be the input
	 * and the reconstructed input with gaussian noise.
	 * This will account for either regularization or none
	 * depending on the configuration.
	 * 
	 * @return reconstruction error
	 */
	public double getReConstructionCrossEntropy() {
		//Matrix preSigH = input.mmul(W).addRowVector(hBias);
		Matrix preSigH = MatrixUtils.addRowVector( this.trainingDataset.times(this.connectionWeights), this.hiddenBiasNeurons.viewRow(0) );
		Matrix sigH = MatrixUtils.sigmoid(preSigH);

		Matrix preSigV = MatrixUtils.addRowVector( sigH.times(this.connectionWeights.transpose()), this.visibleBiasNeurons.viewRow(0) );
		Matrix sigV = MatrixUtils.sigmoid(preSigV);
		Matrix inner = 
				this.trainingDataset.times(MatrixUtils.log(sigV))
				.plus(MatrixUtils.oneMinus( this.trainingDataset )
						.times(MatrixUtils.log(MatrixUtils.oneMinus(sigV))));
		
		//double l = inner.length;
		double l = MatrixUtils.length(inner);
		
		if (this.useRegularization) {
			double normalized = l + l2RegularizedCoefficient();
			double ret = - MatrixUtils.mean( MatrixUtils.rowSums( inner ) ) / normalized;
			
            
            if (this.normalizeByInputRows) {
                ret /= this.trainingDataset.numRows();
            }
            return ret;
			
			
		}
		
		//return - MatrixUtils.mean( MatrixUtils.rowSums( inner ) );
		
        double ret =  - MatrixUtils.mean( MatrixUtils.rowSums( inner ) );
        if (this.normalizeByInputRows) {
        	ret /= this.trainingDataset.numRows();
        }

        return ret;
		
		
	}	
	
    @Override
    public boolean normalizeByInputRows() {
        return normalizeByInputRows;
    }
	
	
	@Override
	public double l2RegularizedCoefficient() {
		//return (MatrixFunctions.pow(getW(),2).sum()/ 2.0)  * l2;
		
		return ( MatrixUtils.sum( MatrixUtils.pow( this.getConnectionWeights(), 2 ) ) / 2.0 ) * l2;
		
	}

    @Override
    public AdagradLearningRate gethBiasAdaGrad() {
        return hBiasAdaGrad;
    }
    @Override
    public void setHbiasAdaGrad(AdagradLearningRate adaGrad) {
        this.hBiasAdaGrad = adaGrad;
    }
    @Override
    public AdagradLearningRate getVBiasAdaGrad() {
        return this.vBiasAdaGrad;
    }
    @Override
    public void setVBiasAdaGrad(AdagradLearningRate adaGrad) {
        this.vBiasAdaGrad = adaGrad;
    }
	
	
	protected void initWeights()  {
		
		if (this.numberVisibleNeurons < 1) {
			throw new IllegalStateException("Number of visible can not be less than 1");
		}
		
		if (this.numberHiddenNeurons < 1) {
			throw new IllegalStateException("Number of hidden can not be less than 1");
		}
		
		
		/*
		 * Initialize based on the number of visible units..
		 * The lower bound is called the fan in
		 * The outer bound is called the fan out.
		 * 
		 * Below's advice works for Denoising AutoEncoders and other 
		 * neural networks you will use due to the same baseline guiding principles for
		 * both RBMs and Denoising Autoencoders.
		 * 
		 * Hinton's Guide to practical RBMs:
		 * The weights are typically initialized to small random values chosen from a zero-mean Gaussian with
		 * a standard deviation of about 0.01. Using larger random values can speed the initial learning, but
		 * it may lead to a slightly worse final model. Care should be taken to ensure that the initial weight
		 * values do not allow typical visible vectors to drive the hidden unit probabilities very close to 1 or 0
		 * as this significantly slows the learning.
		 */
		
		if (this.connectionWeights == null) {
			
			NormalDistribution u = new NormalDistribution( this.randNumGenerator, 0, .01, NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY );

			this.connectionWeights = new DenseMatrix( this.numberVisibleNeurons, this.numberHiddenNeurons );// Matrix.zeros(nVisible,nHidden);
			this.connectionWeights.assign(0.0);

		//	for(int i = 0; i < this.W.rows; i++) 
		//		this.W.putRow(i,new Matrix(u.sample(this.W.columns)));
			
			for ( int i = 0; i < this.connectionWeights.numRows(); i++ ) {
				
				// u.sample( "number of cols in weights" )
				
				double[] rowSamples = u.sample( this.connectionWeights.numCols() );
				
				this.connectionWeights.viewRow(i).assign(rowSamples);
				
				
			}

		}
		
		this.wAdagrad = new AdagradLearningRate( this.connectionWeights.numRows(), this.connectionWeights.numCols() );


		//if(this.hBias == null) {
		if ( this.hiddenBiasNeurons == null) {
			
			//this.hBias = Matrix.zeros(nHidden);
			this.hiddenBiasNeurons = new DenseMatrix(1, this.numberHiddenNeurons);// Matrix.zeros(nHidden);
			this.hiddenBiasNeurons.assign(0.0);
			
			/*
			 * Encourage sparsity.
			 * See Hinton's Practical guide to RBMs
			 */
			//this.hBias.subi(4);
		}
		
		this.hBiasAdaGrad = new AdagradLearningRate( this.hiddenBiasNeurons.numRows(), this.hiddenBiasNeurons.numCols() );
		

		if (this.visibleBiasNeurons == null) {
			
			if (this.trainingDataset != null) {
		
				this.visibleBiasNeurons = new DenseMatrix(1, this.numberVisibleNeurons); // Matrix.zeros(nVisible);
				this.visibleBiasNeurons.assign(0.0);


			} else {
//				this.vBias = Matrix.zeros(nVisible);
				this.visibleBiasNeurons = new DenseMatrix(1, this.numberVisibleNeurons); // Matrix.zeros(nVisible);
				this.visibleBiasNeurons.assign(0.0);
				
			}
			
		}
		
		this.vBiasAdaGrad = new AdagradLearningRate( this.visibleBiasNeurons.numRows(), this.visibleBiasNeurons.numCols() );
		



	}
	
	@Override
	public void clearWeights() {
		
	//	this.connectionWeights = new DenseMatrix( this.numberVisibleNeurons, this.numberHiddenNeurons );// Matrix.zeros(nVisible,nHidden);
		this.connectionWeights.assign(0.0);
		
	//	this.hiddenBiasNeurons = new DenseMatrix(1, this.numberHiddenNeurons);// Matrix.zeros(nHidden);
		this.hiddenBiasNeurons.assign(0.0);
		
	//	this.visibleBiasNeurons = new DenseMatrix(1, this.numberVisibleNeurons); // Matrix.zeros(nVisible);
		this.visibleBiasNeurons.assign(0.0);
		
		this.wAdagrad = new AdagradLearningRate( this.connectionWeights.numRows(), this.connectionWeights.numCols() );

		
	}


	@Override
	public void setRenderEpochs(int renderEpochs) {
		this.renderWeightsEveryNumEpochs = renderEpochs;

	}
	@Override
	public int getRenderEpochs() {
		return renderWeightsEveryNumEpochs;
	}

	@Override
	public double fanIn() {
		return fanIn < 0 ? 1 / this.numberVisibleNeurons : fanIn;
	}

	@Override
	public void setFanIn(double fanIn) {
		this.fanIn = fanIn;
	}

	public void regularize() {
	
		//this.W.addi(W.mul(0.01));
		//this.W.divi(this.momentum);
		
		this.connectionWeights = this.connectionWeights.plus( this.connectionWeights.times(0.01) );
		this.connectionWeights = this.connectionWeights.divide( this.momentum );

	}

	public void scaleWeights( double scale ) {
		
		this.connectionWeights = this.connectionWeights.times( scale );
		
	}
	
	/**
	 * Performs a (parameter average) network merge in the form of
	 * a += b - a / n
	 * where a is a matrix here
	 * b is a matrix on the incoming network
	 * and n is the batch size
	 * @param network the network to merge with
	 * @param batchSize the batch size (number of training examples)
	 * to average by
	 */	
	@Override
	public void merge(NeuralNetworkVectorized network,int batchSize) {

		//			W.addi(network.getW().mini(W).div(batchSize));
		
		MatrixUtils.addi( this.connectionWeights, ( network.getConnectionWeights().minus( this.connectionWeights ).divide(batchSize) ) );
		
//			hBias.addi(network.gethBias().subi(hBias).divi(batchSize));

		MatrixUtils.addi( this.hiddenBiasNeurons, ( network.getHiddenBias().minus( this.hiddenBiasNeurons ).divide( batchSize ) ) );
		
		//			vBias.addi(network.getvBias().subi(vBias).divi(batchSize));

		MatrixUtils.addi( this.visibleBiasNeurons, ( network.getVisibleBias().minus( this.visibleBiasNeurons ).divide( batchSize ) ) );
		
	}
	
	public void jostleWeighMatrix() {
		/*
		 * Initialize based on the number of visible units..
		 * The lower bound is called the fan in
		 * The outer bound is called the fan out.
		 * 
		 * Below's advice works for Denoising AutoEncoders and other 
		 * neural networks you will use due to the same baseline guiding principles for
		 * both RBMs and Denoising Autoencoders.
		 * 
		 * Hinton's Guide to practical RBMs:
		 * The weights are typically initialized to small random values chosen from a zero-mean Gaussian with
		 * a standard deviation of about 0.01. Using larger random values can speed the initial learning, but
		 * it may lead to a slightly worse final model. Care should be taken to ensure that the initial weight
		 * values do not allow typical visible vectors to drive the hidden unit probabilities very close to 1 or 0
		 * as this significantly slows the learning.
		 */
		NormalDistribution u = new NormalDistribution( this.randNumGenerator, 0, .01, fanIn() );

		Matrix weights = new DenseMatrix( this.numberVisibleNeurons, this.numberHiddenNeurons ); //Matrix.zeros(nVisible,nHidden);
		weights.assign(0.0);

		for (int i = 0; i < this.connectionWeights.numRows(); i++) { 
		
			// TODO: figure out whats going on with the weights matrix
		//	weights.putRow(i,new Matrix(u.sample(this.W.columns)));
			
		}




	}	
	
	@Override
	public AdagradLearningRate getAdaGrad() {
		return this.wAdagrad;
	}
	@Override
	public void setAdaGrad(AdagradLearningRate adaGrad) {
		this.wAdagrad = adaGrad;
	}
	
    @Override
    public boolean isUseAdaGrad() {
        return this.useAdaGrad;
    }


    @Override
    public boolean isUseRegularization() {
        return this.useRegularization;
    }
	
	
	
	
}
