package tv.floe.metronome.datasets;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import tv.floe.metronome.classification.neuralnetworks.iterativereduce.mnist.MNIST_DatasetUtils;
import tv.floe.metronome.io.records.MetronomeRecordFactory;
import tv.floe.metronome.types.Pair;

public class UCIDatasets {
	
	public static Pair<Matrix, Matrix> getCovTypeDataset( int max_rows, int num_classes ) throws Exception {

		String LOCAL_DIR_NAME = "CovType";
		String trainingFilesFilename = "covtype.data.gz";
		String trainingFilesFilename_unzipped = "covtype.data";
		String trainingFilesURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz";
				
		String path_to_training_files = "/tmp/" + LOCAL_DIR_NAME + "/" + trainingFilesFilename_unzipped;
		
		DatasetUtils.downloadAndUntar(LOCAL_DIR_NAME, trainingFilesFilename, trainingFilesURL); 

//		MNIST_DatasetUtils util = new MNIST_DatasetUtils( "/tmp/" + LOCAL_DIR_NAME + "/" + trainingFileLabelsFilename_unzipped, "/tmp/" + LOCAL_DIR_NAME + "/" + trainingFilesFilename_unzipped );
//		util.scanIDXFilesAndGenerateInputAndLabels(rowLimit);

		// need to conver the records to MNE
		
//		MetronomeRecordFactory rec_factory = new MetronomeRecordFactory("i:4 | o:3");
	
			
			
//			Vector v_in_0 = new RandomAccessSparseVector(rec_factory.getInputVectorSize());
//			Vector v_out_0 = new RandomAccessSparseVector(rec_factory.getOutputVectorSize());
			
			
			BufferedReader br = new BufferedReader(new FileReader( path_to_training_files ));
			
			int input_cols = 54;
			int output_cols = 1;
			String col_delimiter = ",";
			
			System.out.println("Building CovType Training set as Matricies");
			
			// get row count
			int row_count = 0;
			String line;
			while ((line = br.readLine()) != null) {
				row_count++;
			}
			br.close();
			
			System.out.println("Found " + row_count + " records in " + LOCAL_DIR_NAME + " dataset.");
			
			//int max_rows = 100;
			
			System.out.println("Using " + max_rows + " records in " + LOCAL_DIR_NAME + " dataset.");

			Matrix input_matrix = new DenseMatrix( max_rows, input_cols );
			Matrix output_matrix = new DenseMatrix( max_rows, num_classes );
			
			int row_num = 0;
			int vec_count = 0;
			//String line;
			br.close();
			br = new BufferedReader(new FileReader( path_to_training_files ));
			while ((line = br.readLine()) != null) {

				String[] parts = line.split(col_delimiter);
				if (parts.length != (input_cols + output_cols)) {
					throw new Exception("invalid covtype record in vectorizer!");
				}
				
				
				
				if (row_num % 1000 == 0) {
				
					//System.out.println( "> line > " + line );
					
					for ( int x = 0; x < parts.length; x++ ) {
					
						if ( x < input_cols ) {
							
							input_matrix.viewRow( vec_count ).set(x, Double.parseDouble( parts[ x ] ));
							
						} else {
						
							output_matrix.viewRow( vec_count ).assign(0);
							output_matrix.viewRow( vec_count ).set( Integer.parseInt( parts[ x ] ) - 1, 1.0 );
						}
					}
					
					vec_count++;
				}

				
				
				row_num++;
				
				if ( vec_count >= max_rows ) {
					break;
				}
				
			}
			br.close();	
			
			return new Pair< Matrix, Matrix >( input_matrix, output_matrix );
				
		
		
		
		
		
		
	}
	

}
