package tv.floe.metronome.deeplearning.datasets;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tv.floe.metronome.deeplearning.datasets.fetchers.MnistFetcher;
import tv.floe.metronome.math.ArrayUtils;
import tv.floe.metronome.math.MathUtils;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;


public abstract class DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(DeepLearningTest.class);
/*
	public static Pair<Matrix,Matrix> getIris() throws IOException {
		Pair<Matrix,Matrix> pair = IrisUtils.loadIris();
		return pair;
	}
	public static Pair<Matrix,Matrix> getIris(int num) throws IOException {
		Pair<Matrix,Matrix> pair = IrisUtils.loadIris(num);
		return pair;
	}
	*/
	
	
	
	/**
	 * LFW Dataset: pick first num faces
	 * @param num
	 * @return
	 * @throws Exception
	 */
/*	public static Pair<Matrix,Matrix> getFaces(int num) throws Exception {
		LFWLoader loader = new LFWLoader();
		loader.getIfNotExists();
		return loader.getAllImagesAsMatrix(num);
	}
	*/
	
	/**
	 * LFW Dataset: pick all faces
	 * @param num
	 * @return
	 * @throws Exception
	 */
/*	public static Pair<Matrix,Matrix> getFacesMatrix() throws Exception {
		LFWLoader loader = new LFWLoader();
		loader.getIfNotExists();
		return loader.getAllImagesAsMatrix();
	}
	*/
	
	
	/**
	 * LFW Dataset: pick first num faces
	 * @param num
	 * @return
	 * @throws Exception
	 */
/*	public static List<Pair<Matrix,Matrix>> getFirstFaces(int num) throws Exception {
		LFWLoader loader = new LFWLoader();
		loader.getIfNotExists();
		return loader.getFirst(num);
	}
	*/
	
	/**
	 * LFW Dataset: pick all faces
	 * @param num
	 * @return
	 * @throws Exception
	 */
/*	public List<Pair<Matrix,Matrix>> getFaces() throws Exception {
		LFWLoader loader = new LFWLoader();
		loader.getIfNotExists();
		return loader.getImagesAsList();
	}
	*/
	
	
	/**
	 * Gets an mnist example as an input, label pair.
	 * Keep in mind the return matrix for out come is a 1x1 matrix.
	 * If you need multiple labels, remember to convert to a zeros
	 * with 1 as index of the label for the output training vector.
	 * @param example the example to get
	 * @return the image,label pair
	 * @throws IOException
	 */
	public static Pair<Matrix,Matrix> getMnistExample(int example) throws IOException {
		File ensureExists = new File("/tmp/MNIST");
		if(!ensureExists.exists())
			new MnistFetcher().downloadAndUntar();

		MnistManager man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);
		man.setCurrent(example);
		int[] imageExample = ArrayUtils.flatten(man.readImage());
		return new Pair<Matrix,Matrix>(MatrixUtils.toMatrix(imageExample).transpose(),MatrixUtils.toOutcomeVector(man.readLabel(),10));
	}



	/**
	 * Gets an mnist example as an input, label pair.
	 * Keep in mind the return matrix for out come is a 1x1 matrix.
	 * If you need multiple labels, remember to convert to a zeros
	 * with 1 as index of the label for the output training vector.
	 * @param example the example to get
	 * @param batchSize the batch size of examples to get
	 * @return the image,label pair
	 * @throws IOException
	 */
	public List<Pair<Matrix,Matrix>> getMnistExampleBatches(int batchSize,int numBatches) throws IOException {
		File ensureExists = new File("/tmp/MNIST");
		List<Pair<Matrix,Matrix>> ret = new ArrayList<Pair<Matrix, Matrix>>();
		if(!ensureExists.exists()) 
			new MnistFetcher().downloadAndUntar();
		MnistManager man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);

		int[][] image = man.readImage();
		int[] imageExample = ArrayUtils.flatten(image);

		for(int batch = 0; batch < numBatches; batch++) {
			double[][] examples = new double[batchSize][imageExample.length];
			int[][] outcomeMatrix = new int[batchSize][10];

			for(int i = 1 + batch; i < batchSize + 1 + batch; i++) {
				//1 based indices
				man.setCurrent(i);
				double[] currExample = ArrayUtils.flatten(ArrayUtils.toDouble(man.readImage()));
				examples[i - 1 - batch] = currExample;
				outcomeMatrix[i - 1 - batch] = ArrayUtils.toOutcomeArray(man.readLabel(), 10);
			}
			//Matrix training = new Matrix(examples);
			Matrix training = new DenseMatrix( batchSize, imageExample.length );
			training.assign(examples);
			
			ret.add(new Pair<Matrix, Matrix>(training,MatrixUtils.toMatrix(outcomeMatrix)));
		}

		return ret;
	}

	/**
	 * Gets an mnist example as an input, label pair.
	 * Keep in mind the return matrix for out come is a 1x1 matrix.
	 * If you need multiple labels, remember to convert to a zeros
	 * with 1 as index of the label for the output training vector.
	 * @param example the example to get
	 * @param batchSize the batch size of examples to get
	 * @return the image,label pair
	 * @throws IOException
	 */
	public static Pair<Matrix,Matrix> getMnistExampleBatch(int batchSize) throws IOException {
		File ensureExists = new File("/tmp/MNIST");
		if(!ensureExists.exists() || !new File("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped).exists() || !new File("/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped).exists()) 
			new MnistFetcher().downloadAndUntar();
		MnistManager man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);

		int[][] image = man.readImage();
		int[] imageExample = ArrayUtils.flatten(image);
		double[][] examples = new double[batchSize][imageExample.length];
		int[][] outcomeMatrix = new int[batchSize][10];
		for(int i = 1; i < batchSize + 1; i++) {
			//1 based indices
			man.setCurrent(i);
			double[] currExample = ArrayUtils.flatten(ArrayUtils.toDouble(man.readImage()));
			for(int j = 0; j < currExample.length; j++)
				currExample[j] = MathUtils.normalize(currExample[j], 0, 255);
			examples[i - 1] = currExample;
			outcomeMatrix[i - 1] = ArrayUtils.toOutcomeArray(man.readLabel(), 10);
		}
		Matrix training = new DenseMatrix( batchSize, imageExample.length ); //Matrix(examples);
		training.assign(examples);
		
		return new Pair<Matrix,Matrix>(training,MatrixUtils.toMatrix(outcomeMatrix));

	}


}


