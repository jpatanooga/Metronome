package tv.floe.metronome.deeplearning.datasets.fetchers;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.datasets.MnistManager;
import tv.floe.metronome.math.ArrayUtils;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;


/**
 * Data fetcher for the MNIST dataset
 * @author Adam Gibson
 *
 */
public class MnistDataFetcher extends BaseDataFetcher {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private transient MnistManager man;
	public final static int NUM_EXAMPLES = 60000;

	public MnistDataFetcher() throws IOException {
		if(!new File("/tmp/mnist").exists())
			new MnistFetcher().downloadAndUntar();
		man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);
		numOutcomes = 10;
		totalExamples = NUM_EXAMPLES;
		//1 based cursor
		cursor = 1;
		man.setCurrent(cursor);
		int[][] image;
		try {
			image = man.readImage();
		} catch (IOException e) {
			throw new IllegalStateException("Unable to read image");
		}
		inputColumns = ArrayUtils.flatten(image).length;


	}

	@Override
	public void fetch(int numExamples) {
		if(!hasMore())
			throw new IllegalStateException("Unable to get more; there are no more images");



		//we need to ensure that we don't overshoot the number of examples total
		List<Pair<Matrix,Matrix>> toConvert = new ArrayList<Pair<Matrix,Matrix>>();

		for(int i = 0; i < numExamples; i++,cursor++) {
			if(!hasMore())
				break;
			if(man == null) {
				try {
					man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);
				} catch (IOException e) {
					throw new RuntimeException(e);
				}
			}
			man.setCurrent(cursor);
			//note data normalization
			try {
				Matrix in = MatrixUtils.toMatrix(ArrayUtils.flatten(man.readImage()));
				
				for (int d = 0; d < in.numCols(); d++) {
					
					if (in.get(0, d) > 30) {
						
						in.set(0, d, 1);
						
					} else {
						
						in.set(0, d, 0);
						
					}
					
				}


				Matrix out = createOutputVector(man.readLabel());
				boolean found = false;
				
				for (int col = 0; col < out.numCols(); col++) {
					
					if (out.get(0, col) > 0) {
						found = true;
						break;
					}
					
				}
				if(!found)
					throw new IllegalStateException("Found a matrix without an outcome");

				toConvert.add(new Pair<Matrix, Matrix>(in,out));
			} catch (IOException e) {
				throw new IllegalStateException("Unable to read image");

			}
		}


		initializeCurrFromList(toConvert);



	}

	@Override
	public void reset() {
		cursor = 1;
	}





}