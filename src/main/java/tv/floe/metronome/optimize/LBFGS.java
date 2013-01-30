package tv.floe.metronome.optimize;

import java.util.LinkedList;
/*
import cc.mallet.optimize.BackTrackLineSearch;
import cc.mallet.optimize.InvalidOptimizableException;
import cc.mallet.optimize.LineOptimizer;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.OptimizationException;
import cc.mallet.optimize.OptimizerEvaluator;
import cc.mallet.types.MatrixOps;
*/

/**
 * This is a prototype LBFGS implementation to understand the mechanics of the technique
 * 
 * adapted heavily from MALLET's implementation
 * 
 * 
 * Current Questions
 * 
 * - where are the loops?
 * 
 * - how is input data pulled in?
 * 
 * @author josh
 *
 */
public class LBFGS {


	boolean converged = false;
	
	
	// this function is used with BackTrackLineSearch
	Optimizable.ByGradientValue optimizable;
	
	
	final int maxIterations = 1000;	
	
	// xxx need a more principled stopping point
	//final double tolerance = .0001;
	private double tolerance = .0001;
	final double gradientTolerance = .001;
	final double eps = 1.0e-5;

	// The number of corrections used in BFGS update
	// ideally 3 <= m <= 7. Larger m means more cpu time, memory.
	final int num_corrections = 4; // m

	// Line search function
	private LineOptimizer.ByGradient lineMaximizer;
	
	public LimitedMemoryBFGS (Optimizable.ByGradientValue function) {
		this.optimizable = function;
		lineMaximizer = new BackTrackLineSearch (function);
	}
	
	public Optimizable getOptimizable () { return this.optimizable; }
	public boolean isConverged () { return converged; }


	/**
	 * Sets the LineOptimizer.ByGradient to use in L-BFGS optimization.
	 * @param lineOpt line optimizer for L-BFGS
	 */
	public void setLineOptimizer(LineOptimizer.ByGradient lineOpt) {
		lineMaximizer = lineOpt;
	}

	// State of search
	// g = gradient
	// s = list of m previous "parameters" values
	// y = list of m previous "g" values
	// rho = intermediate calculation
	double [] gradient;
	double [] older_gradient;
	double [] direction;
	double [] parameters;
	double [] older_parameters;
	
	LinkedList saved_parameter_list = new LinkedList(); // s
	LinkedList saved_gradient_list = new LinkedList(); // y
	LinkedList rho = new LinkedList();
	
	double [] alpha;
	static double step = 1.0;
	int iterations;

	private OptimizerEvaluator.ByGradient eval = null;

	// CPAL - added this
	public void setTolerance(double newtol) {
		this.tolerance = newtol;
	}

	public void setEvaluator (OptimizerEvaluator.ByGradient eval) { this.eval = eval; }
	
	public int getIteration () {
		return iterations;
	}
	
	
	public boolean optimize ()
	{
		return optimize (Integer.MAX_VALUE);
	}

	/**
	 * NOTES for the core optimize function
	 * 
	 * 
	 * 
	 * 
	 * 
	 * 
	 * @param numIterations
	 * @return
	 */
	public boolean optimize (int numIterations)
	{

		double initialValue = optimizable.getValue();
		//logger.fine("Entering L-BFGS.optimize(). Initial Value="+initialValue);		


		/**
		 * ########################################################
		 * SETUP
		 * ########################################################
		 * 
		 */
		if(gradient == null) { //first time through
			
			//logger.fine("First time through L-BFGS");
			iterations = 0;
			saved_parameter_list = new LinkedList();
			saved_gradient_list = new LinkedList();
			rho = new LinkedList();
			alpha = new double[num_corrections];	    
			
			// ### initialize alpha ###
			for(int i = 0; i < num_corrections; i++) {
			
				alpha[i] = 0.0;
				
			}

			parameters = new double[optimizable.getNumParameters()];
			older_parameters = new double[optimizable.getNumParameters()];
			
			gradient = new double[optimizable.getNumParameters()];
			older_gradient = new double[optimizable.getNumParameters()];
			direction = new double[optimizable.getNumParameters()];

			optimizable.getParameters (parameters);
			System.arraycopy (parameters, 0, older_parameters, 0, parameters.length);

			optimizable.getValueGradient (gradient);
			System.arraycopy (gradient, 0, older_gradient, 0, gradient.length);
			System.arraycopy (gradient, 0, direction, 0, gradient.length);

			if (MatrixOps.absNormalize (direction) == 0) {
//				logger.info("L-BFGS initial gradient is zero; saying converged");
				gradient = null;
				converged = true;
				return true;
			}
//			logger.fine ("direction.2norm: " + MatrixOps.twoNorm (direction));
			MatrixOps.timesEquals(direction, 1.0 / MatrixOps.twoNorm(direction));
			// make initial jump
//			logger.fine ("before initial jump: \ndirection.2norm: " +
					MatrixOps.twoNorm (direction) + " \ngradient.2norm: " +
					MatrixOps.twoNorm (gradient) + "\nparameters.2norm: " +
					MatrixOps.twoNorm(parameters));

			/**
			 * ########################################################
			 * Optimize the first step with the lineMaximizer
			 * ########################################################
			 */
			//TestMaximizable.testValueAndGradientInDirection (maxable, direction);
			step = lineMaximizer.optimize(direction, step);
			if (step == 0.0) {// could not step in this direction.
				// give up and say converged.
				gradient = null; // reset search
				step = 1.0;
				throw new OptimizationException("Line search could not step in the current direction. " +
						"(This is not necessarily cause for alarm. Sometimes this happens close to the maximum," +
						" where the function may be very flat.)");
			
				//return false;
			}
			
			optimizable.getParameters (parameters);
			optimizable.getValueGradient(gradient);
			
			//logger.fine ("after initial jump: \ndirection.2norm: " +
					MatrixOps.twoNorm (direction) + " \ngradient.2norm: "
					+ MatrixOps.twoNorm (gradient));		
		}
		// ------------ END of SETUP ----------------------

		
		
		/**
		 * ########################################################
		 * ITERATION LOOP
		 * ########################################################
		 */		
		for(int iterationCount = 0; iterationCount < numIterations; iterationCount++)	{
		
			double value = optimizable.getValue();
			
/*			logger.fine("L-BFGS iteration="+iterationCount
					+", value="+value+" g.twoNorm: "+MatrixOps.twoNorm(g)+
					" oldg.twoNorm: "+MatrixOps.twoNorm(oldg));
*/			
			
			
			// get difference between previous 2 gradients and parameters
			double sy = 0.0;
			double yy = 0.0;
			
			// scan through each of the older parameters
			for (int i = 0; i < older_parameters.length; i++) {
				
				// -inf - (-inf) = 0; inf - inf = 0
				if (Double.isInfinite(parameters[i]) &&
						Double.isInfinite(older_parameters[i]) &&
						(parameters[i] * older_parameters[i] > 0)) {
					
					older_parameters[i] = 0.0;
					
				} else {
				
					older_parameters[i] = parameters[i] - older_parameters[i];
					
				}
				
				
				if (Double.isInfinite(gradient[i]) &&
						Double.isInfinite(older_gradient[i]) &&
						(gradient[i] * older_gradient[i] > 0)) {
					
					older_gradient[i] = 0.0;
					
				} else  {
					
					older_gradient[i] = gradient[i] - older_gradient[i];
					
				}
				
				
				
				sy += older_parameters[i] * older_gradient[i]; 	 // si * yi
				
				yy += older_gradient[i] * older_gradient[i];
				
				direction[i] = gradient[i];
			}

			if ( sy > 0 ) {
				throw new InvalidOptimizableException ("sy = "+sy+" > 0" );
			}

			double gamma = sy / yy;	 // scaling factor
			if ( gamma>0 )
				throw new InvalidOptimizableException ("gamma = "+gamma+" > 0" );

			/**
			 * ########################################################
			 * save state of:
			 * rho - intermediate work
			 * older_parameters
			 * older_gradient
			 * ########################################################
			 */
			push (rho, 1.0/sy);
			push (saved_parameter_list, older_parameters);
			push (saved_gradient_list, older_gradient);
			
			
			/**
			 * ########################################################
			 * Calc new direction
			 * ########################################################
			 */			
			// calculate new direction
			assert (saved_parameter_list.size() == saved_gradient_list.size()) :
				"saved_parameter.size: " + saved_parameter_list.size() + " saved_gradient_list.size: " + saved_gradient_list.size();
			
			for(int i = saved_parameter_list.size() - 1; i >= 0; i--) {
				
				alpha[i] =  ((Double)rho.get(i)).doubleValue() *
				MatrixOps.dotProduct ( (double[])saved_parameter_list.get(i), direction);
				
				MatrixOps.plusEquals (direction, (double[])saved_gradient_list.get(i), 
						-1.0 * alpha[i]);
			}
			
			MatrixOps.timesEquals(direction, gamma);
			
			for(int i = 0; i < saved_gradient_list.size(); i++) {
				double beta = (((Double)rho.get(i)).doubleValue()) *
				MatrixOps.dotProduct((double[])saved_gradient_list.get(i), direction);
				MatrixOps.plusEquals(direction,(double[])saved_parameter_list.get(i),
						alpha[i] - beta);
			}

			for (int i=0; i < older_gradient.length; i++) {
				
				older_parameters[i] = parameters[i];
				older_gradient[i] = gradient[i];
				direction[i] *= -1.0;
				
			}
			
/*			logger.fine ("before linesearch: direction.gradient.dotprod: "+
					MatrixOps.dotProduct(direction,g)+"\ndirection.2norm: " +
					MatrixOps.twoNorm (direction) + "\nparameters.2norm: " +
					MatrixOps.twoNorm(parameters));
					*/
			
			/**
			 * ########################################################
			 * Perform line search to get step
			 * ########################################################
			 */			
			//TestMaximizable.testValueAndGradientInDirection (maxable, direction);
			step = lineMaximizer.optimize(direction, step);
			
			
			/**
			 * ########################################################
			 * Can't take a step in this direction
			 * ########################################################
			 */			
			if (step == 0.0) { // could not step in this direction. 
				gradient = null; // reset search
				step = 1.0;
				// xxx Temporary test; passed OK
//				TestMaximizable.testValueAndGradientInDirection (maxable, direction);
				throw new OptimizationException("Line search could not step in the current direction. " +
						"(This is not necessarily cause for alarm. Sometimes this happens close to the maximum," +
						" where the function may be very flat.)");
				//	return false;
			}
			
			optimizable.getParameters (parameters);
			
			optimizable.getValueGradient(gradient);
			
//			logger.fine ("after linesearch: direction.2norm: " +
//					MatrixOps.twoNorm (direction));
			
			double newValue = optimizable.getValue();

			// Test for terminations
			if(2.0*Math.abs(newValue-value) <= tolerance*
					(Math.abs(newValue)+Math.abs(value) + eps)){
				logger.info("Exiting L-BFGS on termination #1:\nvalue difference below tolerance (oldValue: " + value + " newValue: " + newValue);
				converged = true;
				return true;
			}
			
			
			double gg = MatrixOps.twoNorm( gradient );
			
			if(gg < gradientTolerance) {
				logger.fine("Exiting L-BFGS on termination #2: \ngradient="+gg+" < "+gradientTolerance);
				converged = true;
				return true;
			}	    
			
			if(gg == 0.0) {
				logger.fine("Exiting L-BFGS on termination #3: \ngradient==0.0");
				converged = true;
				return true;
			}
			
			logger.fine("Gradient = "+gg);
			
			iterations++;
			
			if (iterations > maxIterations) {
				System.err.println("Too many iterations in L-BFGS.java. Continuing with current parameters.");
				converged = true;
				return true;
				//throw new IllegalStateException ("Too many iterations.");
			}

			//end of iteration. call evaluator
			if (eval != null && !eval.evaluate (optimizable, iterationCount)) {
				logger.fine ("Exiting L-BFGS on termination #4: evaluator returned false.");
				converged = true;
				return false;
			}
		}
		
		return false;
		
	}

	/** Resets the previous gradients and values that are used to
	 * approximate the Hessian. NOTE - If the {@link Optimizable} object
	 * is modified externally, this method should be called to avoid
	 * IllegalStateExceptions. */
	public void reset () {
		gradient = null;
	}

	/**
	 * Pushes a new object onto the queue l
	 * @param l linked list queue of Matrix obj's
	 * @param toadd matrix to push onto queue
	 */
	private void push(LinkedList l, double[] toadd) {
		assert(l.size() <= num_corrections);
		if(l.size() == num_corrections) {
			// remove oldest matrix and add newset to end of list.
			// to make this more efficient, actually overwrite
			// memory of oldest matrix

			// this overwrites the oldest matrix
			double[] last = (double[]) l.get(0);
			System.arraycopy(toadd, 0, last, 0, toadd.length);
			Object ptr = last;
			// this readjusts the pointers in the list
			for(int i=0; i<l.size()-1; i++) 
				l.set(i, (double[])l.get(i+1));			
			l.set(m-1, ptr);
		}
		else {
			double [] newArray = new double[toadd.length];
			System.arraycopy (toadd, 0, newArray, 0, toadd.length);
			l.addLast(newArray);
		}
	}

	/**
	 * Pushes a new object onto the queue l
	 * @param l linked list queue of Double obj's
	 * @param toadd double value to push onto queue
	 */
	private void push(LinkedList l, double toadd) {
		assert(l.size() <= num_corrections);
		if(l.size() == num_corrections) { //pop old double and add new
			l.removeFirst(); 
			l.addLast(new Double(toadd));
		}
		else 
			l.addLast(new Double(toadd));
	}
	
	
	
	
}
