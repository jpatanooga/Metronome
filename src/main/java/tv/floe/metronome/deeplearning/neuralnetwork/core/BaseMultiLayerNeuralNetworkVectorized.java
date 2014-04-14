package tv.floe.metronome.deeplearning.neuralnetwork.core;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



import tv.floe.metronome.classification.neuralnetworks.core.NeuralNetwork;
import tv.floe.metronome.deeplearning.math.transforms.MatrixTransform;
import tv.floe.metronome.deeplearning.neuralnetwork.activation.ActivationFunction;
import tv.floe.metronome.deeplearning.neuralnetwork.activation.Activations;
import tv.floe.metronome.deeplearning.neuralnetwork.core.NeuralNetworkVectorized.LossFunction;
import tv.floe.metronome.deeplearning.neuralnetwork.core.NeuralNetworkVectorized.OptimizationAlgorithm;
import tv.floe.metronome.deeplearning.neuralnetwork.gradient.LogisticRegressionGradient;
import tv.floe.metronome.deeplearning.neuralnetwork.gradient.MultiLayerGradient;
import tv.floe.metronome.deeplearning.neuralnetwork.gradient.NeuralNetworkGradient;
import tv.floe.metronome.deeplearning.neuralnetwork.layer.HiddenLayer;
import tv.floe.metronome.deeplearning.neuralnetwork.optimize.MultiLayerNetworkOptimizer;
import tv.floe.metronome.deeplearning.neuralnetwork.serde.Persistable;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

public abstract class BaseMultiLayerNeuralNetworkVectorized implements Serializable,Persistable {

    /**
     *
     */
    private static final long serialVersionUID = 4066891298715416874L;


    private static Logger log = LoggerFactory.getLogger( BaseMultiLayerNeuralNetworkVectorized.class );


    public int inputNeuronCount;
    public int outputNeuronCount;

    //the hidden layer sizes at each layer
    public int[] hiddenLayerSizes;
    public int numberLayers;

    //the hidden layers
    public HiddenLayer[] hiddenLayers;
    public LogisticRegression logisticRegressionLayer;
    // DA / RBM Layers
    public NeuralNetworkVectorized[] preTrainingLayers;

    public RandomGenerator randomGenerator;
    public RealDistribution distribution;

    // the input data ---- how is this going to be handled?
    // how was it handled with the OOP-MLPN version?
    public Matrix inputTrainingData = null;
    public Matrix outputTrainingLabels = null;

    public double learningRateUpdate = 0.95;
    public boolean useRegularization = true;
    public double l2 = 0.01;
    private double momentum = 0.1;
    //don't use sparsity by default
    private double sparsity = 0;

    //optional: used in normalizing input. This is used in saving the model for prediction purposes in normalizing incoming data
    private Matrix columnSums = null;
    //subtract input by column means for zero mean
    private Matrix  columnMeans = null;
    //divide by the std deviation
    private Matrix columnStds = null;


    public MultiLayerNetworkOptimizer optimizer;

    protected Map<Integer,MatrixTransform> weightTransforms = new HashMap<Integer,MatrixTransform>();

    //hidden bias transforms; for initialization
    private Map<Integer,MatrixTransform> hiddenBiasTransforms = new HashMap<Integer,MatrixTransform>();
    //visible bias transforms for initialization
    private Map<Integer,MatrixTransform> visibleBiasTransforms = new HashMap<Integer,MatrixTransform>();

    private boolean useAdaGrad = false;
    private boolean shouldBackProp = true;
    //whether to only train a certain number of epochs
    private boolean forceNumEpochs = false;

    private boolean initCalled = false;
    private boolean useHiddenActivationsForwardProp = true;

    /*
     * The delta from the previous iteration to this iteration for
     * cross entropy must change >= this amount in order to continue.
     */
    public double errorTolerance = 0.0001;

    /*
     * Drop out: randomly zero examples
     */
    protected double dropOut = 0;

    /*
     * Normalize by input rows with gradients or not
     */
    protected boolean normalizeByInputRows = false;

    /**
     * Which optimization algorithm to use: SGD or CG
     */
    protected OptimizationAlgorithm optimizationAlgorithm;
    /**
     * Which loss function to use:
     * Squared loss, Reconstruction entropy, negative log likelihood
     */
    protected LossFunction lossFunction;


    /**
     * CTOR
     *
     */
    public BaseMultiLayerNeuralNetworkVectorized() {

    }

    public BaseMultiLayerNeuralNetworkVectorized(int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers, RandomGenerator rng) {

        this(n_ins,hidden_layer_sizes,n_outs,n_layers,rng,null,null);

    }


    public BaseMultiLayerNeuralNetworkVectorized(int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers, RandomGenerator rng, Matrix input, Matrix labels) {

        this.inputNeuronCount = n_ins;
        this.hiddenLayerSizes = hidden_layer_sizes;
        this.inputTrainingData = input;
        this.outputTrainingLabels = labels;

        if(hidden_layer_sizes.length != n_layers) {
            throw new IllegalArgumentException("The number of hidden layer sizes must be equivalent to the nLayers argument which is a value of " + n_layers);
        }

        this.outputNeuronCount = n_outs;
        this.numberLayers = n_layers;

        this.hiddenLayers = new HiddenLayer[n_layers];
        this.preTrainingLayers = createNetworkLayers( this.numberLayers );

        if (rng == null) {
            this.randomGenerator = new MersenneTwister(123);
        } else {
            this.randomGenerator = rng;
        }


        if (input != null) {
            initializeLayers(input);
        }



    }

    /* sanity check for hidden layer and inter layer dimensions */
    private void dimensionCheck() {

        for (int i = 0; i < this.numberLayers; i++) {

            HiddenLayer h = this.hiddenLayers[i];
            NeuralNetworkVectorized network = this.preTrainingLayers[i];

            //h.getW().assertSameSize(network.getW());
            MatrixUtils.assertSameLength( h.connectionWeights, network.getConnectionWeights() );
//            h.getB().assertSameSize(network.gethBias());
            MatrixUtils.assertSameLength( h.biasTerms, network.getHiddenBias() );


            if (i < this.numberLayers - 1) {


                HiddenLayer h1 = this.hiddenLayers[ i + 1 ];
                NeuralNetworkVectorized network1 = this.preTrainingLayers[ i + 1 ];

                if ( h1.getNeuronInputCount() != h.getNeuronOutputCount() ) {
                    throw new IllegalStateException("Invalid structure: hidden layer in for " + (i + 1) + " not equal to number of ins " + i);
                }

                if (network.getnHidden() != network1.getnVisible()) {
                    throw new IllegalStateException("Invalid structure: network hidden for " + (i + 1) + " not equal to number of visible " + i);
                }

            }
        }

        if ( this.hiddenLayers[ this.hiddenLayers.length - 1 ].getNeuronOutputCount() != this.logisticRegressionLayer.getnIn() ) {
            throw new IllegalStateException("Number of outputs for final hidden layer not equal to the number of logistic input units for output layer");
        }


    }

    /**
     * Synchronizes the rng, this is mean for use with scale out methods
     */
/*    public void synchonrizeRng() {
        RandomGenerator rgen = new SynchronizedRandomGenerator(rng);
        for(int i = 0; i < nLayers; i++) {
            layers[i].setRng(rgen);
            sigmoidLayers[i].setRng(rgen);
        }


    }
*/


    /**
     * Base class for initializing the layers based on the input.
     * This is meant for capturing numbers such as input columns or other things.
     *
     * This method sets up two types of layers:
     * - normal ML-NN layers
     * - RBM / DA layers
     *
     * @param input the input matrix for training
     */
    protected void initializeLayers(Matrix input) {
        Matrix layer_input = input;
        int input_size;

        //System.out.println("initializeLayers --------");

        // construct multi-layer
        for (int i = 0; i < this.numberLayers; i++) {

            if (i == 0) {

                //input_size = this.nIns;
                input_size = this.inputNeuronCount;

                // construct sigmoid_layer
                //this.sigmoidLayers[i] = new HiddenLayer(input_size, this.hiddenLayerSizes[i], null, null, rng,layer_input);
                this.hiddenLayers[ i ] = new HiddenLayer(input_size, this.hiddenLayerSizes[i], this.randomGenerator );
                this.hiddenLayers[ i ].setInput( layer_input );



            } else {

                input_size = this.hiddenLayerSizes[ i - 1 ];
                layer_input = this.hiddenLayers[i - 1].sampleHiddenGivenLastVisible();
                // construct sigmoid_layer
                this.hiddenLayers[ i ] = new HiddenLayer(input_size, this.hiddenLayerSizes[i], this.randomGenerator);
                this.hiddenLayers[ i ].setInput( layer_input );

            }

            // construct DL appropriate class for pre training layer
            this.preTrainingLayers[ i ] = createPreTrainingLayer( layer_input,input_size, this.hiddenLayerSizes[i], this.hiddenLayers[i].connectionWeights, this.hiddenLayers[i].biasTerms, null, this.randomGenerator, i );
        }

        System.out.println("Logistic Output Layer: Inputs: " + this.hiddenLayerSizes[this.numberLayers-1] + ", Output Classes: " + this.outputNeuronCount );

        this.logisticRegressionLayer = new LogisticRegression(layer_input, this.hiddenLayerSizes[this.numberLayers-1], this.outputNeuronCount );

        if ( this.useAdaGrad ) {
            this.logisticRegressionLayer.setUseAdaGrad(true);
        }

        if ( this.normalizeByInputRows ) {
            this.logisticRegressionLayer.setNormalizeByInputRows(true);
        }

        dimensionCheck();
        applyTransforms();
        initCalled = true;


    }

    public synchronized Map<Integer, MatrixTransform> getHiddenBiasTransforms() {

        return hiddenBiasTransforms;

    }

    public synchronized Map<Integer, MatrixTransform> getVisibleBiasTransforms() {

        return visibleBiasTransforms;

    }

    /**
     * Returns the -fanIn to fanIn
     * coefficient used for initializing the
     * weights.
     * The default is 1 / nIns
     * @return the fan in coefficient
     */
/*	public double fanIn() {
		if(this.in < 0)
			return 1.0 / nIns;
		return fanIn;
	}
*/

    public double getReconstructionCrossEntropy() {

        double sum = 0;

        for(int i = 0; i < this.numberLayers; i++) {

            sum += this.preTrainingLayers[i].getReConstructionCrossEntropy();

        }

        sum /= (double) this.numberLayers;
        return sum;
    }



    /**
     * Compute Activations
     * - TODO: considering adding a param to take on inputs
     *
     * @return
     */
    public List<Matrix> feedForward() {

        if (this.inputTrainingData == null) {
            throw new IllegalStateException("Unable to perform feed forward; no input found");
        }

        List<Matrix> activations = new ArrayList<Matrix>();
        Matrix currInput = this.inputTrainingData;
        activations.add( currInput );

        for (int i = 0; i < this.numberLayers; i++) {
/*			
			HiddenLayer layer = this.hiddenLayers[ i ];
			//layers[i].setInput(input);
			this.preTrainingLayers[i].setInput(input);
			input = layer.computeOutputActivation(input);
			activations.add(input);
*/

            //getLayers()[i].setInput(currInput);
            this.preTrainingLayers[i].setInput(currInput);
            //getSigmoidLayers()[i].setInput(input);
            HiddenLayer layer = this.hiddenLayers[ i ];
            layer.setInput( this.inputTrainingData );

            if (useHiddenActivationsForwardProp) {
                //currInput = getSigmoidLayers()[i].activate(currInput);
                currInput = layer.computeOutputActivation(currInput);
            } else {
                //currInput = getLayers()[i].sampleHiddenGivenVisible(currInput).getSecond();
                currInput = this.preTrainingLayers[ i ].sampleHiddenGivenVisible(currInput).getSecond();
            }

            activations.add(currInput);


        }

        this.logisticRegressionLayer.input = currInput;
        activations.add( this.logisticRegressionLayer.predict(currInput) );
        return activations;
    }

    /**
     * TODO: make sure our concept of an activation function can deliver functionality
     *
     * @param deltaRet
     */
    private void computeDeltas(List<Pair<Matrix,Matrix>> deltaRet) {

        Matrix[] gradients = new Matrix[ this.numberLayers + 2 ];
        Matrix[] deltas = new Matrix[ this.numberLayers + 2 ];
        ActivationFunction derivative = this.hiddenLayers[ 0 ].activationFunction;

        ActivationFunction softMaxDerivative = Activations.softmax();

        //- y - h
        Matrix delta = null;

        List<Matrix> activations = this.feedForward();

		/*
		 * Precompute activations and z's (pre activation network outputs)
		 */
        List<Matrix> weights = new ArrayList<Matrix>();

        for (int j = 0; j < this.preTrainingLayers.length; j++) {

            weights.add( this.preTrainingLayers[j].getConnectionWeights() );

        }

        weights.add( this.logisticRegressionLayer.connectionWeights );

/*		
		List<Matrix> zs = new ArrayList<Matrix>();
		
		zs.add( this.inputTrainingData );
		
		for (int i = 0; i < this.preTrainingLayers.length; i++) {
			
			if (this.preTrainingLayers[i].getInput() == null && i == 0) {
				
				this.preTrainingLayers[i].setInput( this.inputTrainingData );
				
			} else if (this.preTrainingLayers[i].getInput() == null) {
				
				this.feedForward();
				
			}

			zs.add(MatrixUtils.sigmoid( MatrixUtils.addRowVector( this.preTrainingLayers[ i ].getInput().times( weights.get( i ) ),  this.preTrainingLayers[i].getHiddenBias().viewRow(0) )));
			
		}
		
		zs.add( MatrixUtils.addRowVector( this.logisticRegressionLayer.input.times( this.logisticRegressionLayer.connectionWeights ), this.logisticRegressionLayer.biasTerms.viewRow(0) ) );
*/

        Matrix labels = this.outputTrainingLabels;

        //errors
        for (int i = this.numberLayers + 1; i >= 0; i--) {

            // output layer
            if (i >= this.numberLayers + 1) {
				/*
				Matrix z = activations.get(i); 
				
				//- y - h
				delta = MatrixUtils.neg( labels.minus( z ) );

				//(- y - h) .* f'(z^l) where l is the output layer
				Matrix initialDelta = MatrixUtils.elementWiseMultiplication( delta, derivative.applyDerivative( z ) );
				deltas[ i ] = initialDelta;
*/

                //-( y - h) .* f'(z^l) where l is the output layer
                //delta = labels.sub(activations.get(i)).neg().mul(softMaxDerivative.applyDerivative(activations.get(i)));
                Matrix tmpDelta = MatrixUtils.neg( labels.minus( activations.get( i ) ) );

                delta = MatrixUtils.elementWiseMultiplication( tmpDelta, softMaxDerivative.applyDerivative(activations.get(i)) );

                deltas[i] = delta;


            } else {

                //derivative i + 1; aka gradient for bias
/*				
				delta = deltas[ i + 1 ];
				Matrix w = weights.get( i ).transpose();
				Matrix z = activations.get( i ); //zs.get( i );
				Matrix a = activations.get( i );
				//W^t * error^l + 1

				Matrix error = delta.times( w );
				deltas[ i ] = error;

				error = MatrixUtils.elementWiseMultiplication( error, derivative.applyDerivative(z) );

				deltas[ i ] = error;
				
				//calculate gradient for layer
				Matrix lastLayerDelta = deltas[i + 1].transpose();
				Matrix newGradient = lastLayerDelta.times(a);
				
				gradients[ i ] = newGradient.divide( this.inputTrainingData.numRows() );
				*/

                //W^t * error^l + 1
                Matrix tmpDeltas = deltas[ i + 1 ].times( weights.get(i).transpose() );

                tmpDeltas = MatrixUtils.elementWiseMultiplication(tmpDeltas, derivative.applyDerivative(activations.get(i)));

                deltas[ i ] = tmpDeltas;

                //calculate gradient for layer
                Matrix newGradient = deltas[ i + 1 ].transpose().times( activations.get( i ) );
                gradients[i] = newGradient;


            }

        }


        for (int i = 0; i < gradients.length; i++) {

            deltaRet.add(new Pair<Matrix, Matrix>(gradients[i],deltas[i]));

        }

    }




    /**
     * Backpropagation of errors for weights
     * @param lr the learning rate to use
     * @param epochs  the number of epochs to iterate (this is already called in finetune)
     */
    public void backProp(double lr,int epochs) {

        double lastEntropy = this.negativeLogLikelihood();

        //store a copy of the network for when binary cross entropy gets
        //worse after an iteration
        BaseMultiLayerNeuralNetworkVectorized revert = this.clone();

        //sgd style; only train a certain number of epochs

        if (this.forceNumEpochs) {

            for (int i = 0; i < epochs; i++) {

                backPropStep( revert, lr, i );
                lastEntropy = this.negativeLogLikelihood();

            }

        } else {


            boolean train = true;
            int count = 0;
            double changeTolerance = 1e-5;
            int backPropIterations = 0;

            while (train) {

                if ( backPropIterations >= epochs ) {
                    log.info("Backprop number of iterations max hit; convering");
                    break;

                }


                count++;
                backPropStep( revert, lr, count );

                this.logisticRegressionLayer.trainTillConvergence( lr, epochs );

                double entropy = this.negativeLogLikelihood();

                if( entropy < lastEntropy ) {
                    double diff = Math.abs(entropy - lastEntropy);
                    if(diff < changeTolerance) {
                        log.info("Not enough of a change on back prop...breaking");
                        break;
                    }
                    else
                        lastEntropy = entropy;
                    log.info("New negative log likelihood " + lastEntropy);
                    revert = clone();
                }

                else if(entropy >= lastEntropy) {
                    train = false;
                }

                backPropIterations++;


            }


        }

    }

    /**
     * just keeping it around for now, for historical perspective
     *
     * @param lr
     * @param epochs
     */
    public void oldBackProp(double lr, int epochs) {


        for (int i = 0; i < epochs; i++) {

            List<Matrix> activations = feedForward();

            //precompute deltas
            List<Pair<Matrix,Matrix>> deltas = new ArrayList<Pair<Matrix, Matrix>>();
            //computeDeltas(activations, deltas);
            computeDeltas( deltas );


            for (int l = 0; l < this.numberLayers; l++) {

                Matrix add = deltas.get( l ).getFirst().divide( this.inputTrainingData.numRows() ).times( lr );
                if(normalizeByInputRows)
                    add = add.divide( this.inputTrainingData.numRows() );

                if (useRegularization) {

                    //add = add.times( this.preTrainingLayers[ l ].getConnectionWeights().times( l2 ) );
                    add = MatrixUtils.elementWiseMultiplication(add, this.preTrainingLayers[ l ].getConnectionWeights().times( l2 ));

                }


                this.preTrainingLayers[ l ].setConnectionWeights( this.preTrainingLayers[ l ].getConnectionWeights().minus( add.times( lr ) ) );

                this.hiddenLayers[ l ].connectionWeights = this.preTrainingLayers[l].getConnectionWeights();
                Matrix deltaColumnSums = MatrixUtils.columnSums( deltas.get( l + 1 ).getSecond() );

                // TODO: check this, needs to happen in place?
                deltaColumnSums = deltaColumnSums.divide( this.inputTrainingData.numRows() );

                // TODO: check this, needs to happen in place?
                //this.preTrainingLayers[ l ].getHiddenBias().subi( deltaColumnSums.times( lr ) );
                Matrix hbiasMinus = this.preTrainingLayers[ l ].getHiddenBias().minus( deltaColumnSums.times( lr ) );
                this.preTrainingLayers[ l ].sethBias(hbiasMinus);

                this.hiddenLayers[ l ].biasTerms = this.preTrainingLayers[l].getHiddenBias();
            }


            this.logisticRegressionLayer.connectionWeights = this.logisticRegressionLayer.connectionWeights.plus(deltas.get( this.numberLayers ).getFirst());


        }





    }

    /**
     * Do a back prop iteration.
     * This involves computing the activations, tracking the last layers weights
     * to revert to in case of convergence, the learning rate being used to train
     * and the current epoch
     * @param revert the best network so far
     * @param lr the learning rate to use for training
     * @param epoch the epoch to use
     * @return whether the training should converge or not
     */
    protected void backPropStep(BaseMultiLayerNeuralNetworkVectorized revert, double lr, int epoch) {
        //feedforward to compute activations
        //initial error


        //precompute deltas
        List<Pair<Matrix,Matrix>> deltas = new ArrayList<Pair<Matrix,Matrix>>();
        //compute derivatives and gradients given activations
        computeDeltas(deltas);


        for (int l = 0; l < this.numberLayers; l++) {

            Matrix add = deltas.get( l ).getFirst().transpose();
            if(normalizeByInputRows)
                add = add.divide( this.inputTrainingData.numRows() );
            //get the gradient
            if (this.useAdaGrad ) {
                add = MatrixUtils.elementWiseMultiplication(add,this.preTrainingLayers[ l ].getAdaGrad().getLearningRates( add ));

            } else {

                //add.muli(lr);
                add = add.times( lr );

            }

            //add.divi(input.rows);
            if(normalizeByInputRows)
                MatrixUtils.divi( add , this.inputTrainingData.numRows() );


            //l2
            if (this.useRegularization) {

                //	add.muli(this.getLayers()[l].getW().mul(l2));
                add = MatrixUtils.elementWiseMultiplication(add,this.preTrainingLayers[ l ].getConnectionWeights().times( l2 ));

            }

            // -------- update W
            //	this.getLayers()[l].getW().addi(add);
            this.preTrainingLayers[ l ].setConnectionWeights(this.preTrainingLayers[ l ].getConnectionWeights().minus( add ));

            //	this.getSigmoidLayers()[l].setW(layers[l].getW());
            this.hiddenLayers[ l ].setWeights( this.preTrainingLayers[ l ].getConnectionWeights() );


            // ---------- update hidden bias
            //	DoubleMatrix deltaColumnSums = deltas.get(l + 1).getSecond().columnSums();
            Matrix deltaColumnSums = MatrixUtils.columnSums( deltas.get( l + 1 ).getSecond() );

            //	deltaColumnSums.divi(input.rows);
            if(normalizeByInputRows)
                MatrixUtils.divi( deltaColumnSums, this.inputTrainingData.numRows() );

            //	getLayers()[l].gethBias().addi(deltaColumnSums.mul(lr));
            MatrixUtils.subi( this.preTrainingLayers[ l ].getHiddenBias(), deltaColumnSums.times( lr ) );
            //	getSigmoidLayers()[l].setB(getLayers()[l].gethBias());
            this.hiddenLayers[ l ].biasTerms = this.preTrainingLayers[ l ].getHiddenBias();

        }



        //logistic regression update (note the transpose below, the deltas come out flipped from their counterparts
        Matrix logLayerGradient = deltas.get(numberLayers).getFirst().transpose();
        Matrix biasGradient = MatrixUtils.columnSums(deltas.get(numberLayers).getSecond());


        if(momentum != 0)
            biasGradient = biasGradient.times(momentum);

        if(useAdaGrad)
            biasGradient = MatrixUtils.elementWiseMultiplication(biasGradient,logisticRegressionLayer.getBiasAdaGrad().getLearningRates(biasGradient));
        else
            biasGradient = biasGradient.times(lr);

        if(normalizeByInputRows)
            biasGradient = biasGradient.divide(inputTrainingData.numRows());


        if(useAdaGrad)
            logLayerGradient = MatrixUtils.elementWiseMultiplication(logLayerGradient,logisticRegressionLayer.getBiasAdaGrad().getLearningRates(logLayerGradient));


        else
            logLayerGradient = logLayerGradient.times(lr);

        if(momentum != 0)
            logLayerGradient = logLayerGradient.times(momentum);


        if(normalizeByInputRows)
            logLayerGradient.divide(this.inputTrainingData.numRows());



        MatrixUtils.subi(logisticRegressionLayer.connectionWeights,logLayerGradient);

        int biasGradientLength = biasGradient.numCols() * biasGradient.numRows();
        int biasTermsLength = logisticRegressionLayer.biasTerms.numRows() * logisticRegressionLayer.biasTerms.numCols();
        if(biasGradientLength != biasTermsLength)
            MatrixUtils.subi(logisticRegressionLayer.biasTerms,MatrixUtils.mean(biasGradient));
        else {
            double mean = MatrixUtils.mean(biasGradient);
            MatrixUtils.subi(logisticRegressionLayer.biasTerms,mean);

        }



    }

    // just a placeholder
    @Override
    public BaseMultiLayerNeuralNetworkVectorized clone() {
        //	BaseMultiLayerNetwork ret = new Builder<>().withClazz(getClass()).buildEmpty();

        BaseMultiLayerNeuralNetworkVectorized ret = null;
        //	ret.update(this);
        return ret;
    }

    /**
     * Gets the multi layer gradient for this network.
     * This includes calculating the gradients for each layer
     * @param params the params to pass (k, corruption level,...)
     * @param lr the learning rate to use for logistic regression
     * @return the multi layer gradient for the whole network
     */
    public MultiLayerGradient getGradient(Object[] params) {

        List<NeuralNetworkGradient> gradient = new ArrayList<NeuralNetworkGradient>();
        for (NeuralNetworkVectorized network : this.preTrainingLayers) {

            gradient.add( network.getGradient(params) );

        }

        double lr = 0.01;

        if (params.length >= 2) {
            lr = (Double) params[1];
        }

        this.feedForward();
        LogisticRegressionGradient g2 = this.logisticRegressionLayer.getGradient(lr);


        MultiLayerGradient ret =  new MultiLayerGradient(gradient,g2);
/*
		if(multiLayerGradientListeners != null && !multiLayerGradientListeners.isEmpty()) {
			for(MultiLayerGradientListener listener : multiLayerGradientListeners) {
				listener.onMultiLayerGradient(ret);
			}
		}
*/
        return ret;
    }


    /**
     * Creates a layer depending on the index.
     * The main reason this matters is for continuous variations such as the {@link CDBN}
     * where the first layer needs to be an {@link CRBM} for continuous inputs
     *
     */
    public abstract NeuralNetworkVectorized createPreTrainingLayer(Matrix input, int nVisible, int nHidden, Matrix weights, Matrix hbias, Matrix vBias, RandomGenerator rng, int index);



    public void finetune(double learningRate, int epochs) {

        finetune( this.outputTrainingLabels, learningRate, epochs );

    }

    /**
     * Run SGD based on the given output vectors
     *
     *
     * @param labels the labels to use
     * @param lr the learning rate during training
     * @param epochs the number of times to iterate
     */
    public void finetune(Matrix outputLabels, double learningRate, int epochs) {

        if (null != outputLabels) {
            this.outputTrainingLabels = outputLabels;
        }

        optimizer = new MultiLayerNetworkOptimizer(this,learningRate);
        optimizer.optimize( outputLabels, learningRate, epochs );
        //optimizer.optimizeWSGD( outputLabels, learningRate, epochs );

    }


    /**
     * Label the probabilities of the input
     * @param x the input to label
     * @return a vector of probabilities
     * given each label.
     *
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     *
     *
     */
    public Matrix predict(Matrix x) {

        Matrix input = x;

        for(int i = 0; i < this.numberLayers; i++) {
            HiddenLayer layer = this.hiddenLayers[i];
            input = layer.computeOutputActivation(input);
        }

        return this.logisticRegressionLayer.predict(input);
    }


    /**
     * Serializes this to the output stream.
     * @param os the output stream to write to
     */
    public void write(OutputStream os) {
        try {
            //	ObjectOutputStream oos = new ObjectOutputStream(os);
            //	oos.writeObject(this);

            //		MatrixWritable.writeMatrix(arg0, arg1)this.hiddenLayers[ 0 ].biasTerms

            //	ByteArrayOutputStream out = new ByteArrayOutputStream();
            DataOutput d = new DataOutputStream(os);

            MatrixWritable.writeMatrix(d, inputTrainingData);
            // d.writeUTF(src_host);
/*		    //d.writeInt(this.SrcWorkerPassCount);
		    d.writeInt(this.GlobalPassCount);
		    
		    d.writeInt(this.IterationComplete);
		    d.writeInt(this.CurrentIteration);
		    
		    d.writeInt(this.TrainedRecords);
		    //d.writeFloat(this.AvgLogLikelihood);
		    d.writeFloat(this.PercentCorrect);
		    d.writeDouble(this.RMSE);
		    */


            //d.write

            // buf.write
            // MatrixWritable.writeMatrix(d, this.worker_gradient.getMatrix());
            //MatrixWritable.writeMatrix(d, this.parameter_vector);
            // MatrixWritable.
            //	    ObjectOutputStream oos = new ObjectOutputStream(out);



        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    public void write( String filename ) throws IOException {

        File file = new File( filename );

        if (!file.exists()) {

            try {
                file.getParentFile().mkdirs();
            } catch (Exception e) {

            }
            file.createNewFile();

        }

        FileOutputStream oFile = new FileOutputStream(filename, false);
        this.write(oFile);
        oFile.close();



    }

    /**
     * Load (using {@link ObjectInputStream}
     * @param is the input stream to load from (usually a file)
     */
    public void load(InputStream is) {
        try {
            ObjectInputStream ois = new ObjectInputStream(is);
            BaseMultiLayerNeuralNetworkVectorized loaded = (BaseMultiLayerNeuralNetworkVectorized) ois.readObject();
            update(loaded);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    /**
     * Load (using {@link ObjectInputStream}
     * @param is the input stream to load from (usually a file)
     */
    public static BaseMultiLayerNeuralNetworkVectorized loadFromFile(InputStream is) {
        try {
            ObjectInputStream ois = new ObjectInputStream(is);
            log.info("Loading network model...");

            BaseMultiLayerNeuralNetworkVectorized loaded = (BaseMultiLayerNeuralNetworkVectorized) ois.readObject();
            return loaded;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    /**
     * Helper method for loading the model file from disk by path name
     *
     * @param filename
     * @return
     * @throws Exception
     */
    public static BaseMultiLayerNeuralNetworkVectorized loadFromFile(String filename ) throws Exception {

        BaseMultiLayerNeuralNetworkVectorized nn = null;

        File file = new File( filename );
        if(!file.exists()) {
            //
            //file.createNewFile();
            throw new Exception("Model File Path does not exist!");
        }

        //FileOutputStream oFile = new FileOutputStream(filename, false);



        try {

            DataInputStream dis = new DataInputStream(
                    new FileInputStream( filename ));

            nn = BaseMultiLayerNeuralNetworkVectorized.loadFromFile( dis );


            //dataOutputStream.flush();
            dis.close();

        } catch (IOException e) {
            log.error("Unable to load model",e);
        }

        return nn;



    }

    /**
     * Assigns the parameters of this model to the ones specified by this
     * network. This is used in loading from input streams, factory methods, etc
     * @param network the network to get parameters from
     */
    protected void update(BaseMultiLayerNeuralNetworkVectorized network) {

        this.preTrainingLayers = new NeuralNetworkVectorized[ network.preTrainingLayers.length ];

        for (int i = 0; i < preTrainingLayers.length; i++) {

            this.preTrainingLayers[i] = network.preTrainingLayers[ i ].clone();

        }

        this.hiddenLayerSizes = network.hiddenLayerSizes;
        this.logisticRegressionLayer = network.logisticRegressionLayer.clone();
        this.logisticRegressionLayer.input = network.logisticRegressionLayer.input;
        this.logisticRegressionLayer.labels = network.logisticRegressionLayer.labels;

        this.inputNeuronCount = network.inputNeuronCount;
        this.numberLayers = network.numberLayers;
        this.outputNeuronCount = network.outputNeuronCount;
        this.randomGenerator = network.randomGenerator;
        this.distribution = network.distribution;
        this.useAdaGrad = network.useAdaGrad;

        if ( null != network.columnMeans ) {
            this.columnMeans.assign( network.columnMeans );
        }

        if ( null != network.columnStds ) {
            this.columnStds.assign( network.columnStds );
        }

        if ( null != network.columnSums ) {
            this.columnSums.assign( network.columnSums );
        }

        this.forceNumEpochs = network.forceNumEpochs;
        this.l2 = network.l2;
        this.momentum = network.momentum;
        this.learningRateUpdate = network.learningRateUpdate;
        this.shouldBackProp = network.shouldBackProp;
        this.sparsity = network.sparsity;


        this.hiddenLayers = new HiddenLayer[network.hiddenLayers.length];

        for (int i = 0; i < hiddenLayers.length; i++) {

            this.hiddenLayers[ i ] = network.hiddenLayers[ i ].clone();

        }

        this.weightTransforms = network.weightTransforms;
        this.visibleBiasTransforms = network.visibleBiasTransforms;
        this.hiddenBiasTransforms = network.hiddenBiasTransforms;


    }

    public void initBasedOn(BaseMultiLayerNeuralNetworkVectorized network) {

        this.update(network);

        // now clear all connections.


        for (int i = 0; i < preTrainingLayers.length; i++) {

            this.preTrainingLayers[i].clearWeights();

        }

        this.logisticRegressionLayer.clearWeights();

        for (int i = 0; i < hiddenLayers.length; i++) {

            this.hiddenLayers[ i ].clearWeights();

        }

    }


    /**
     * @return the negative log likelihood of the model
     */
    public double negativeLogLikelihood() {
        return this.logisticRegressionLayer.negativeLogLikelihood();
    }

    /**
     * Train the network running some unsupervised
     * pretraining followed by SGD/finetune
     * @param input the input to train on
     * @param labels the labels for the training examples(a matrix of the following format:
     * [0,1,0] where 0 represents the labels its not and 1 represents labels for the positive outcomes
     * @param otherParams the other parameters for child classes (algorithm specific parameters such as corruption level for SDA)
     */
    public abstract void trainNetwork(Matrix input,Matrix labels,Object[] otherParams);

    /**
     * Creates a layer depending on the index.
     * The main reason this matters is for continuous variations such as the {@link CDBN}
     * where the first layer needs to be an {@link CRBM} for continuous inputs
     * @param input the input to the layer
     * @param nVisible the number of visible inputs
     * @param nHidden the number of hidden units
     * @param W the weight vector
     * @param hbias the hidden bias
     * @param vBias the visible bias
     * @param rng the rng to use (THiS IS IMPORTANT; YOU DO NOT WANT TO HAVE A MIS REFERENCED RNG OTHERWISE NUMBERS WILL BE MEANINGLESS)
     * @param index the index of the layer
     * @return a neural network layer such as {@link RBM}
     */
//	public abstract NeuralNetwork createLayer(Matrix input,int nVisible,int nHidden, Matrix W,Matrix hbias,Matrix vBias,RandomGenerator rng,int index);


    public abstract NeuralNetworkVectorized[] createNetworkLayers(int numLayers);

    /**
     * Apply transforms to RBMs before we train
     *
     *
     *
     */
    protected void applyTransforms() {

        // do we have RBMs at all
        if(this.preTrainingLayers == null || this.preTrainingLayers.length < 1) {
            throw new IllegalStateException("Layers not initialized");
        }

        for (int i = 0; i < this.preTrainingLayers.length; i++) {

            if (weightTransforms.containsKey(i)) {

                //		layers[i].setW(weightTransforms.get(i).apply(layers[i].getW()));
                this.preTrainingLayers[i].setConnectionWeights( weightTransforms.get(i).apply( this.preTrainingLayers[i].getConnectionWeights() ) );

            }

            if (hiddenBiasTransforms.containsKey(i)) {

                preTrainingLayers[i].sethBias(getHiddenBiasTransforms().get(i).apply(preTrainingLayers[i].getHiddenBias()));

            }

            if (this.visibleBiasTransforms.containsKey(i)) {

                preTrainingLayers[i].setVisibleBias(getVisibleBiasTransforms().get(i).apply(preTrainingLayers[i].getVisibleBias()));

            }




        }

    }

    public synchronized double getMomentum() {
        return momentum;
    }

    public synchronized void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public synchronized Map<Integer, MatrixTransform> getWeightTransforms() {
        return weightTransforms;
    }

    public synchronized void setWeightTransforms(
            Map<Integer, MatrixTransform> weightTransforms) {
        this.weightTransforms = weightTransforms;
    }

    public synchronized void addWeightTransform( int layer,MatrixTransform transform) {

        this.weightTransforms.put(layer,transform);

    }

    public synchronized double getSparsity() {
        return sparsity;
    }

    public synchronized void setSparsity(double sparsity) {
        this.sparsity = sparsity;
    }

    public String generateNetworkSizeReport() {

        String out = "";

        long hiddenLayerConnectionCount = 0;
        long preTrainLayerConnectionCount = 0;

        for ( int x = 0; x < this.numberLayers; x++ ) {

            hiddenLayerConnectionCount += MatrixUtils.length( this.hiddenLayers[ x ].connectionWeights );

        }

        for ( int x = 0; x < this.numberLayers; x++ ) {

            preTrainLayerConnectionCount += MatrixUtils.length( this.preTrainingLayers[ x ].getConnectionWeights() );

        }


        out += "Number of Hidden / RBM Layers: " + this.numberLayers + "\n";
        out += "Total Hidden Layer Connection Count: " + hiddenLayerConnectionCount + "\n";
        out += "Total PreTrain (RBM) Layer Connection Count: " + preTrainLayerConnectionCount + "\n";

        return out;

    }

    public String generateNetworkStateReport() {

        String out = "";

        out += "Number of Hidden / RBM Layers: " + this.numberLayers + "\n";
        out += "inputNeuronCount: " + this.inputNeuronCount + "\n";
        out += "l2: " + this.l2 + "\n";
        out += "learningRateUpdate: " + this.learningRateUpdate + "\n";
        out += "momentum: " + this.momentum + "\n";
        out += "outputNeuronCount: " + this.outputNeuronCount + "\n";
        out += "sparsity: " + this.sparsity + "\n";
        out += "this.hiddenLayers.length: " + this.hiddenLayers.length + "\n";
        out += "this.logisticRegressionLayer.l2: " + this.logisticRegressionLayer.l2 + "\n";
        out += "this.logisticRegressionLayer.nIn: " + this.logisticRegressionLayer.nIn + "\n";
        out += "this.logisticRegressionLayer.nOut: " + this.logisticRegressionLayer.nOut + "\n";
        out += "this.logisticRegressionLayer.useRegularization: " + this.logisticRegressionLayer.useRegularization + "\n";
        out += "this.useRegularization: " + this.useRegularization + "\n";
        //out += "this.useRegularization: " + this. + "\n";

        return out;

    }

    /**
     * Merges this network with the other one.
     * This is a weight averaging with the update of:
     * a += b - a / n
     * where a is a matrix on the network
     * b is the incoming matrix and n
     * is the batch size.
     * This update is performed across the network layers
     * as well as hidden layers and logistic layers
     *
     * @param network the network to merge with
     * @param batchSize the batch size (number of training examples)
     * to average by
     */
    public void merge(BaseMultiLayerNeuralNetworkVectorized network, int batchSize) {

        if (network.numberLayers != this.numberLayers) {

            throw new IllegalArgumentException("Unable to merge networks that are not of equal length");

        }

        for (int i = 0; i < this.numberLayers; i++) {

            // pretrain layers
            NeuralNetworkVectorized n = this.preTrainingLayers[i];
            NeuralNetworkVectorized otherNetwork = network.preTrainingLayers[i];
            n.merge(otherNetwork, batchSize);

            //tied weights: must be updated at the same time
            //getSigmoidLayers()[i].setB(n.gethBias());
            this.hiddenLayers[i].biasTerms = n.getHiddenBias();
            //getSigmoidLayers()[i].setW(n.getW());
            this.hiddenLayers[i].connectionWeights = n.getConnectionWeights();

        }

        //getLogLayer().merge(network.logLayer, batchSize);
        this.logisticRegressionLayer.merge(network.logisticRegressionLayer, batchSize);

    }

    public boolean isForceNumEpochs() {
        return forceNumEpochs;
    }

    public  void setForceNumEpochs(boolean forceNumEpochs) {
        this.forceNumEpochs = forceNumEpochs;
    }

    public boolean isShouldBackProp() {
        return shouldBackProp;
    }





}
