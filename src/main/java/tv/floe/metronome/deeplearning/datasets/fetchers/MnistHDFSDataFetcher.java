package tv.floe.metronome.deeplearning.datasets.fetchers;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.cloudera.iterativereduce.io.TextRecordParser;

import tv.floe.metronome.deeplearning.datasets.MnistManager;
import tv.floe.metronome.deeplearning.datasets.iterator.DataSetFetcher;
import tv.floe.metronome.io.records.MetronomeRecordFactory;
import tv.floe.metronome.io.records.RecordFactory;
import tv.floe.metronome.math.ArrayUtils;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

/**
 * Given that right now I dont want to write a custom new binary record reader
 * we'll convert the MNIST dataset to the metronome format and just let it run as text for now
 * - text files are far more manageable from a known heuristic to "split" in hdfs for workers
 * 
 * 
 * 
 * TODO:
 * - need to setup the Metronome vectorizer to parse the line into a vector
 * 		- replaces the functionality of the MnistManager
 * 
 * - need to collect and cache the lines from the data block locally
 * - need to setup a mechanism to batch out records from the cache into a vectorized format
 * 		- 
 * 
 * 
 * Down the Road
 * - when I get a bit more time I'd like to make a version that reads the binary format
 * 		*not on the critical path right now*
 * 
 * @author josh
 *
 */
public class MnistHDFSDataFetcher extends BaseDataFetcher implements DataSetFetcher {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	//private transient MnistManager man;
	public final static int NUM_EXAMPLES = 60000;

	int currentVectorIndex = 0;
	
	TextRecordParser record_reader = null;
	RecordFactory vector_factory = null;
	
	// tells the record factory how to layout the vectors in|out
	private String vectorSchema = "i:784 | o:10";

	boolean bCacheIsHot = false;

	/**
	 * For now we'll just give the fetcher an instantiated parser
	 * - may can be a more elegant way to do this
	 * 
	 * We're assuming a text input format with a Metronome vector layout
	 * 
	 * @param hdfsLineParser
	 * @throws IOException
	 */
	public MnistHDFSDataFetcher( TextRecordParser hdfsLineParser ) throws IOException {

		this.record_reader = hdfsLineParser;
		this.vector_factory = new MetronomeRecordFactory( this.vectorSchema );
		
/*		
		if (!new File("/tmp/mnist").exists()) {
			new MnistFetcher().downloadAndUntar();
		}
	*/	
		//man = new MnistManager( "/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped, "/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped );
		numOutcomes = 10;
		//totalExamples = NUM_EXAMPLES;
		//1 based cursor
		cursor = 1;
		//man.setCurrent(cursor);
		//int[][] image;
/*
		try {
			image = man.readImage();
		} catch (IOException e) {
			throw new IllegalStateException("Unable to read image");
		}
*/
		inputColumns = 784; //ArrayUtils.flatten(image).length;


	}

	/**
	 * Converts a line of Metronome record format to the Pair<Image,Label> format expected by the dataset code
	 * 
	 * expects the data to be in the raw state from the binary image files
	 * 
	 * TODO:
	 * - look at efficiency here, prolly need to let the vector factory convert straight to Matrix recs
	 * 
	 * @param line
	 * @return
	 */
	public Pair<Matrix,Matrix> convertMetronomeTextLineToMatrixInputPair( String line ) {
		
		Vector v_in = new RandomAccessSparseVector( this.vector_factory.getFeatureVectorSize());
		Vector v_out = new RandomAccessSparseVector( this.vector_factory.getOutputVectorSize());
		
		//rec_factory.vectorizeLine(record_0, v_in, v_out);
		try {
			this.vector_factory.vectorizeLine(line, v_in, v_out);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		Matrix input = new DenseMatrix( 1, v_in.size() );
		input.viewRow(0).assign(v_in);
		
		// normalize
		for (int d = 0; d < input.numCols(); d++) {
			
			if (input.get(0, d) > 30) {
				
				input.set(0, d, 1);
				
			} else {
				
				input.set(0, d, 0);
				
			}
			
		}		
		
		Matrix label = new DenseMatrix( 1, v_out.size() );
		label.viewRow(0).assign(v_out);
		
		//Matrix out = createOutputVector(man.readLabel());
		boolean found = false;
		
		for (int col = 0; col < label.numCols(); col++) {
			
			if (label.get(0, col) > 0) {
				found = true;
				break;
			}
			
		}
		
		if (!found) {
			
			throw new IllegalStateException("Found a matrix without an outcome");
			
		}
		
		
		return new Pair<Matrix, Matrix>(input, label);
		
	}

	/**
	 * NOTE:
	 * 
	 * - be sure to preserve the data normalization
	 * 
	 * TODO:
	 * - cache the read vectors into batches
	 * 		- do we cache the batch or just let that get recreated?
	 * 
	 */
	@Override
	public void fetch(int numExamples) {
		
	    Text value = new Text();	    
	    boolean result = true;
		
		
		//if (!hasMore()) {
		if (false == this.record_reader.hasMoreRecords()) {
			
			throw new IllegalStateException("Unable to get more; there are no more images");
			
		}

		// so here we replace the MnistManager with the Hadoop based record reader to start pulling text based
		// lines off hdfs

		// so we need to convert each line into a 1 row matrix along w a 1 row matrix for the label

		//we need to ensure that we don't overshoot the number of examples total
		List<Pair<Matrix,Matrix>> toConvert = new ArrayList<Pair<Matrix,Matrix>>();

		for (int i = 0; i < numExamples; i++, cursor++ ) {
			
			if (false == this.record_reader.hasMoreRecords()) {
				
				break;
				
			}
			
			try {
				result = this.record_reader.next(value);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			toConvert.add( this.convertMetronomeTextLineToMatrixInputPair( value.toString() ));
		}


		initializeCurrFromList(toConvert);



	}

	@Override
	public void reset() {
		cursor = 1;
	}

}
