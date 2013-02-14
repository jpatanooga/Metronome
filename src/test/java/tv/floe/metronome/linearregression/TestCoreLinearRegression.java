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

	// lr: 0.003
	//private static String file_name = "src/test/resources/linear_regression_sample_data.txt";
	
	// lr: 3
	private static String file_name = "src/test/resources/sat_scores_svmlight.txt";
	//private static String file_name = "src/test/resources/two_points.txt";
	//private static String file_name = "src/test/resources/big_points.txt";
	//private static String file_name = "src/test/resources/normalized_points.txt";
	
	
	
	@Test
	public void test() throws Exception {
		
		ParallelOnlineLinearRegression polr = new ParallelOnlineLinearRegression(
				2, new UniformPrior()).alpha(1)
				.stepOffset(1000).decayExponent(0.9).lambda(3.0e-5)
				.learningRate(17);
		
		RCV1RecordFactory factory = new RCV1RecordFactory();

		
		for ( int x = 0; x < 3000; x++ ) {
			
			
			BufferedReader reader = new BufferedReader(new FileReader(file_name));
			double error_sum = 0;
			int rec_count = 0;
			
			String line = reader.readLine();
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


/*					String[] ar = line.split(" |f 0:");
					//System.out.println(ar.length);
					System.out.println( ar[2] + " |f 0:" + ar[0] );
	*/			    
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
		

	}

}
