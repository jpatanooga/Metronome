package tv.floe.metronome.linearregression;

import static org.junit.Assert.assertEquals;

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

public class TestRegressionStatistics {
	
	private static String file_name = "src/test/resources/data/SAT_Scores/sat_scores_svmlight.txt";
	
	
	
	@Test
	public void TestBaseRSquaredCalc( ) throws Exception {
		

		ParallelOnlineLinearRegression polr = new ParallelOnlineLinearRegression(
				2, new UniformPrior()).alpha(1)
				.stepOffset(1000).decayExponent(0.9).lambda(3.0e-5)
				.learningRate(17);
		
		RCV1RecordFactory factory = new RCV1RecordFactory();
		
		
		RegressionStatistics regStats = new RegressionStatistics();
		
		BufferedReader reader = new BufferedReader(new FileReader(file_name));
		double error_sum = 0;
		int rec_count = 0;
		
		double y_partial_sum = 0;
		
		String line = reader.readLine();
		while (line != null && line.length() > 0) {

			
			if (null == line || line.trim().equals("")) {
				
			} else {

				System.out.println( "> " + line  );
	
				Vector vec = new RandomAccessSparseVector(2);
				
			    
			    double y_observed = factory.processLineAlt(line, vec);
				
			    System.out.println( "Parsed: y:" + y_observed + ", x: " + vec.get(1) + "\n" );
			    
			    y_partial_sum += y_observed;
			    rec_count++;
			    
				
			} // if
			
			
			line = reader.readLine();
			
			
		} // while
		
		assertEquals( y_partial_sum, 11850, 0.0 );
		assertEquals( rec_count, 20 );
		
	    regStats.AddPartialSumForY(y_partial_sum, rec_count);
		
	    double y_bar = regStats.ComputeYAvg();
	    
	    assertEquals( y_bar, 592.5, 0.0 );
	    
	    
	    // do the learning pass ------------------------------------------
	    
	    reader.close();

		for ( int x = 0; x < 3000; x++ ) {
			
			
			reader = new BufferedReader(new FileReader(file_name));
			error_sum = 0;
			rec_count = 0;
			
			line = reader.readLine();
			while (line != null && line.length() > 0) {
	
				//System.out.println(line);
	
				
	
				if (null == line || line.trim().equals("")) {
					
				} else {
					
					rec_count++;
	
					Vector vec = new RandomAccessSparseVector(2);
					
				    
				    double actual = factory.processLineAlt(line, vec);

				    //Utils.PrintVector(vec);
					
				    // we're only looking at the first row or the matrix because 
				    // the original code was for multinomial log regression
				    // but here we only need a single parameter vector
				    double hypothesis_value = polr.getBeta().viewRow(0).dot(vec);
				    
				    double error = Math.abs( hypothesis_value - actual ); // SquaredErrorLossFunction.Calc(hypothesis_value, actual);
				    error_sum += error;
				    
				    polr.train(actual, vec);

				}
				
				line = reader.readLine();

			} // while
			
			//" + error_sum + " / " + rec_count + " = 
			System.out.println("> " + x + " Avg Err: " + ( error_sum / (rec_count) ) );
			
			// reader.reset();
			System.out.println("----------------------- ");			
		} // for
		
		    System.out.print( "beta: ");
		    Utils.PrintVector(polr.getBeta().viewRow(0));	    
	    
		    reader.close();
	    
	    // now simulate the post-pass ------------------------------------
	    
	    double SSyy_partial_sum = 0;
	    double SSE_partial_sum = 0;
	    
	    reader = new BufferedReader(new FileReader(file_name));
	    
	    line = reader.readLine();
		while (line != null && line.length() > 0) {

			
			if (null == line || line.trim().equals("")) {
				
			} else {

				Vector vec = new RandomAccessSparseVector(2);
				
			    
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
		
		System.out.println( "R-Squared: " + r_squared );
	    
		assertEquals( r_squared, 0.8588903008127426, 0.0 );
		
	}
	
	

}
