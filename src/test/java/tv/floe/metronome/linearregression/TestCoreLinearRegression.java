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
 * @author josh
 * 
 */
public class TestCoreLinearRegression {

	private static String file_name = "src/test/resources/linear_regression_sample_data.txt";
	//private static String file_name = "src/test/resources/two_points.txt";
	//private static String file_name = "src/test/resources/big_points.txt";
	//private static String file_name = "src/test/resources/normalized_points.txt";
	
	
	
	@Test
	public void test() throws Exception {
		
		ParallelOnlineLinearRegression polr = new ParallelOnlineLinearRegression(
				2, new UniformPrior()).alpha(1)
				.stepOffset(1000).decayExponent(0.9).lambda(3.0e-5)
				.learningRate(.0003);
		
		RCV1RecordFactory factory = new RCV1RecordFactory();

		
		for ( int x = 0; x < 10; x++ ) {
			
			
			BufferedReader reader = new BufferedReader(new FileReader(file_name));
			double error_sum = 0;
			
			
			String line = reader.readLine();
			while (line != null && line.length() > 0) {
	
				//System.out.println(line);
	
				
	
				if (null == line || line.trim().equals("")) {
					
				} else {
	
					Vector vec = new RandomAccessSparseVector(2);
				    
				    double actual = factory.processLineNew(line, vec);
					
				    // we're only looking at the first row or the matrix because 
				    // the original code was for multinomial log regression
				    // but here we only need a single parameter vector
				    double hypothesis_value = polr.getBeta().viewRow(0).dot(vec);
				    
				//    System.out.println( "hyp: " + hypothesis_value );
				    //System.out.println( "error: " + (hypothesis_value - actual) );
				    
				    
				  //  System.out.print( "vec: ");
				 //   Utils.PrintVector(vec);
				    
				    double error = SquaredErrorLossFunction.Calc(hypothesis_value, actual);
				    error_sum += error;
				    
				    //System.out.println("> Err: " + error ); // + " \t#h: " + hypothesis_value + " \ta: " + actual );
				    
//				    System.out.println(">> Vec: " + vec.get(0) );
	
				    
				    
				    polr.train(actual, vec);
				    
				    //System.out.println("#\n");
					
//				    System.out.println(">> Beta: " + polr.getBeta().viewRow(0).get(0) );
				    
				}
				
				line = reader.readLine();

			} // while
			
			System.out.println("> " + x + " Avg Err: " + ( error_sum / (x+1) ) );
			
			// reader.reset();
			System.out.println("----------------------- ");			
		} // for
		
		    System.out.print( "beta: ");
		    Utils.PrintVector(polr.getBeta().viewRow(0));
		

	}

}
