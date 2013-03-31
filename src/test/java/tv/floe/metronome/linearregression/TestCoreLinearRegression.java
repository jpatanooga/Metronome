package tv.floe.metronome.linearregression;

import static org.junit.Assert.*;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import org.apache.mahout.classifier.sgd.UniformPrior;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import tv.floe.metronome.io.records.RCV1RecordFactory;
import tv.floe.metronome.utils.Utils;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;

/**
 * Run the sample data from the resources folder through the LR process - hand
 * check the resultant equation
 * 
 * Current Issues: 
 * - needs a bias term
 * 
 * 
 * 
 * http://mathbits.com/MathBits/TIsection/Statistics2/linear.htm
 * 
 * @author josh
 * 
 */
public class TestCoreLinearRegression {
	

	

//	QuickTestConfig qtf = new QuickTestConfig("src/test/resources/data/SAT_Scores/sat_scores_svmlight.txt", 17, 3000, 2 );

//	QuickTestConfig qtf = new QuickTestConfig("src/test/resources/data/R_synth_data_10032013_v1.csv", 0.0002, 1000, 2 ); // breaks R-Square somehow? large X-values? [bug]	
	
	// beta:  0.09412438653921235, 4.051704052081415 ----- Final R-Squared: 0.9962182306267638
	//QuickTestConfig qtf = new QuickTestConfig("src/test/resources/data/temp/temp.txt", 0.02, 3000, 2 );
	
//	QuickTestConfig qtf = new QuickTestConfig("src/test/resources/data/synth/R_synth_20_10p5_var4.csv", 20, 300, 2 );

//	QuickTestConfig qtf = new QuickTestConfig("src/test/resources/data/synth/R_synth_20_10p5_var10.csv", 20, 300, 2 );
	
//	QuickTestConfig qtf = new QuickTestConfig("src/test/resources/data/synth/R_synth_2_10_var2.csv", 20, 300, 2 );

	// R_m4_synth_20_10p5_var4.csv
//	QuickTestConfig qtf = new QuickTestConfig("src/test/resources/R/R_m4_synth_20_10p5_var4.csv", 20, 300, 2 );
	
	//R_synth_multi_4coef_test.csv
	QuickTestConfig qtf = new QuickTestConfig("src/test/resources/data/synth/R_synth_multi_4coef_test.csv", 5, 10, 5 );
	
	@Test
	public void test() throws Exception {
		
		ParallelOnlineLinearRegression polr = new ParallelOnlineLinearRegression(
				qtf.feature_size, new UniformPrior()).alpha(1)
				.stepOffset(1000).decayExponent(0.9).lambda(3.0e-5)
				.learningRate(qtf.learningRate);
		
		RCV1RecordFactory factory = new RCV1RecordFactory();


		double y_partial_sum = 0;
		double y_bar = 0;
	    double SSyy_partial_sum = 0;
	    double SSE_partial_sum = 0;
	    
	    
	    
		
		for ( int x = 0; x < qtf.iterations; x++ ) {
			
			RegressionStatistics regStats = new RegressionStatistics();
			
			BufferedReader reader = new BufferedReader(new FileReader(qtf.fileName));
			double error_sum = 0;
			int rec_count = 0;
			
		    SSyy_partial_sum = 0;
		    SSE_partial_sum = 0;
			
			
			String line = reader.readLine();
			while (line != null && line.length() > 0) {
	
				//System.out.println(line);
	
				
	
				if (null == line || line.trim().equals("")) {
					
				} else {
					
					rec_count++;
	
					Vector vec = new RandomAccessSparseVector(qtf.feature_size);
					
				    
				    double actual = factory.processLineAlt(line, vec);

				    // we're only looking at the first row or the matrix because 
				    // the original code was for multinomial log regression
				    // but here we only need a single parameter vector
				    double hypothesis_value = polr.getBeta().viewRow(0).dot(vec);
				    
				    double error = Math.abs( hypothesis_value - actual ); // SquaredErrorLossFunction.Calc(hypothesis_value, actual);
				    error_sum += error;

					polr.train(actual, vec);

				    // now calc Regression Stats stuff ----
				    
				    if ( x == 0 ) {
				    	
				    	// calc the avg stuff
				    	y_partial_sum += actual;
				    	
				    } else {
				    	
				    	// calc the ongoing r-squared
					    SSyy_partial_sum += Math.pow( (actual - y_bar), 2 );
					    
					    SSE_partial_sum += Math.pow( (actual - hypothesis_value), 2 );
				    	
				    	
				    }

				} // if
				
				line = reader.readLine();

			} // while
			
			System.out.println("> " + x + " Avg Err: " + ( error_sum / (rec_count) ) );

			// setup the avg'd y-bar data
		    if ( x == 0 ) {
		    	
		    	System.out.println( "y-sum: " + y_partial_sum + ", rec-count: " + rec_count );
		    	
		    	regStats.AddPartialSumForY(y_partial_sum, rec_count);
		    	y_bar = regStats.ComputeYAvg();
		    	
		    	System.out.println( "y-bar: " + y_bar );
		    	
		    } else {
		    	
				regStats.AccumulateSSEPartialSum(SSE_partial_sum);
				regStats.AccumulateSSyyPartialSum(SSyy_partial_sum);
			    
				double r_squared = regStats.CalculateRSquared();
				
				System.out.println( "> " + x + " R-Squared: " + r_squared );
		    	
		    	
		    }
			
			System.out.println("----------------------- ");			
		} // for
		
		    System.out.print( "beta: ");
		    Utils.PrintVector(polr.getBeta().viewRow(0));
		
		    // now simulate the post-pass ------------------------------------
		    
		    SSyy_partial_sum = 0;
		    SSE_partial_sum = 0;
		    
			RegressionStatistics regStats = new RegressionStatistics();
			
			BufferedReader reader = new BufferedReader(new FileReader(qtf.fileName));
		    
		    String line = reader.readLine();
			while (line != null && line.length() > 0) {

				
				if (null == line || line.trim().equals("")) {
					
				} else {
		
					Vector vec = new RandomAccessSparseVector(qtf.feature_size);
				    
				    double y_observed = factory.processLineAlt(line, vec);
				    double y_predicted = polr.getBeta().viewRow(0).dot(vec);
				    
				    SSyy_partial_sum += Math.pow( (y_observed - y_bar), 2 );
				    
				    SSE_partial_sum += Math.pow( (y_observed - y_predicted), 2 );
				    
					
				} // if
				
				
				line = reader.readLine();
				
				
			} // while
				    
			regStats.AccumulateSSEPartialSum(SSE_partial_sum);
			regStats.AccumulateSSyyPartialSum(SSyy_partial_sum);
			
			reader.close();
		    
			double r_squared = regStats.CalculateRSquared();
			
			System.out.println( "Final R-Squared: " + r_squared );		    

			polr.Debug();

	}

}
