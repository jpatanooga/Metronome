package tv.floe.metronome.deeplearning.dbn.util;

import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.dbn.DeepBeliefNetwork;
import tv.floe.metronome.math.MatrixUtils;

public class DBNDebuggingUtil {
	
	public static void printMatrixSection(Matrix m, int rowLimit, int colLimit) {
		
		
		for (int r = 0; r < rowLimit; r++) {
			for ( int c = 0; c < colLimit; c++ ) {
	
				System.out.print(" " + m.get(r, c));
				
			}
			System.out.println(" ");
		}		
		
	}
	
	public static void printDebugLayers( DeepBeliefNetwork dbn, int elementLimitPerLayer ) {
		
		System.out.println( "print DBN debug stuff ----------------- " );
		
		for ( int x = 0; x < dbn.preTrainingLayers.length; x++ ) {
		
			System.out.println( "Pre Train Layer: " );
			printMatrixSection( dbn.preTrainingLayers[ x ].getConnectionWeights(), 2, 2 ); 

		}
		
		for ( int x = 0; x < dbn.hiddenLayers.length; x++ ) {
			
			System.out.println( "Normal Layer: " );
			printMatrixSection( dbn.hiddenLayers[ x ].connectionWeights, 2, 2 ); 

		}
		
		System.out.println( "Logistic Layer: " );
		printMatrixSection( dbn.logisticRegressionLayer.connectionWeights, 2, 2 );
		
		
	}

}
