package tv.floe.metronome.deeplearning.datasets;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

import com.google.common.collect.Lists;

/**
 * A data set (example/outcome pairs)
 * The outcomes are specifically for neural network encoding such that
 * any labels that are considered true are 1s. The rest are zeros.
 * @author Adam Gibson
 *
 */
public class DataSet extends Pair<Matrix, Matrix> {

	private static final long serialVersionUID = 1935520764586513365L;
	private static Logger log = LoggerFactory.getLogger(DataSet.class);

	public DataSet(Pair<Matrix,Matrix> pair) {
		this(pair.getFirst(),pair.getSecond());
	}

	public DataSet(Matrix first, Matrix second) {
		super(first, second);
	}

	public DataSet copy() {
		return new DataSet(getFirst(),getSecond());
	}
	
	public static DataSet merge(List<DataSet> data) {
		
		if (data.isEmpty()) {
		
			throw new IllegalArgumentException("Unable to merge empty dataset");
			
		}
		
		DataSet first = data.iterator().next();

		Matrix inputData = new DenseMatrix( data.size(), first.getFirst().numCols() );
		Matrix outputLabels = new DenseMatrix( data.size(), first.getSecond().numCols() );

		
		// samples.assignRow(x, input.viewRow(x) );

		for (int i = 0; i < data.size(); i++) {
			
			//in.putRow(i,data.get(i).getFirst());
			inputData.assignRow( i, data.get(i).getFirst().viewRow(0) );
			//out.putRow(i,data.get(i).getSecond());
			outputLabels.assignRow( i, data.get(i).getSecond().viewRow(0) );

		}
		
		return new DataSet( inputData, outputLabels );
		
	}

	public void validate() {
		if(getFirst().numRows() != getSecond().numRows())
			throw new IllegalStateException("Invalid dataset");
	}
	
	public int outcome() {
		if(this.numExamples() > 1)
			throw new IllegalStateException("Unable to derive outcome for dataset greater than one row");
		//return SimpleBlas.iamax(getSecond());
		return MatrixUtils.iamax( getSecond().viewRow(0) );
	}

	public DataSet get(int i) {
		//return new DataSet(getFirst().getRow(i),getSecond().getRow(i));
		
		Matrix inputRow = getFirst().viewPart(i, 1, 0, getFirst().numCols());
		Matrix labelRow = getSecond().viewPart(i, 1, 0, getSecond().numCols());
		
		return new DataSet( inputRow, labelRow );
	}
	
	public List<List<DataSet>> batchBy(int num) {
		
		return Lists.partition(asList(),num);
		
	}
	
	
	

	/**
	 * Sorts the dataset by label:
	 * Splits the data set such that examples are sorted by their labels.
	 * A ten label dataset would produce lists with batches like the following:
	 * x1   y = 1
	 * x2   y = 2
	 * ...
	 * x10  y = 10
	 * @return a list of data sets partitioned by outcomes
	 */
	public List<List<DataSet>> sortAndBatchByNumLabels() {
		sortByLabel();
		return Lists.partition(asList(),numOutcomes());
	}

	public List<List<DataSet>> batchByNumLabels() {
		return Lists.partition(asList(),numOutcomes());
	}


	public List<DataSet> asList() {
		
		List<DataSet> list = new ArrayList<DataSet>(numExamples());
		
		for (int i = 0; i < numExamples(); i++)  {
			
			//list.add(new DataSet(getFirst().getRow(i),getSecond().getRow(i)));
			
			//Matrix inputRow = getFirst().viewPart(i, 1, 0, getFirst().numCols());
			Matrix inputRow = MatrixUtils.viewRowAsMatrix( getFirst(), i );
			//Matrix labelRow = getSecond().viewPart(i, 1, 0, getSecond().numCols());
			Matrix labelRow = MatrixUtils.viewRowAsMatrix( getSecond(), i );
			
			list.add( new DataSet( inputRow, labelRow ) );
			
			
		}
		
		return list;
	}

	public Pair<DataSet,DataSet> splitTestAndTrain(int numHoldout) {

		if(numHoldout >= numExamples())
			throw new IllegalArgumentException("Unable to split on size larger than the number of rows");


		List<DataSet> list = asList();

		Collections.rotate(list, 3);
		Collections.shuffle(list);
		List<List<DataSet>> partition = new ArrayList<List<DataSet>>();
		partition.add(list.subList(0, numHoldout));
		partition.add(list.subList(numHoldout, list.size()));
		DataSet train = merge(partition.get(0));
		DataSet test = merge(partition.get(1));
		return new Pair<DataSet, DataSet>(train,test);
	}

	/**
	 * Organizes the dataset to minimize sampling error
	 * while still allowing efficient batching.
	 */
	public void sortByLabel() {
		Map<Integer,Queue<DataSet>> map = new HashMap<Integer,Queue<DataSet>>();
		List<DataSet> data = asList();
		int numLabels = numOutcomes();
		int examples = numExamples();
		for(DataSet d : data) {
			int label = getLabel(d);
			Queue<DataSet> q = map.get(label);
			if(q == null) {
				q = new ArrayDeque<DataSet>();
				map.put(label, q);
			}
			q.add(d);
		}

		for(Integer label : map.keySet()) {
			log.info("Label " + label + " has " + map.get(label).size() + " elements");
		}

		//ideal input splits: 1 of each label in each batch
		//after we run out of ideal batches: fall back to a new strategy
		boolean optimal = true;
		for(int i = 0; i < examples; i++) {
			if(optimal) {
				for(int j = 0; j < numLabels; j++) {
					Queue<DataSet> q = map.get(j);
					DataSet next = q.poll();
					//add a row; go to next
					if(next != null) {
						addRow(next,i);
						i++;
					}
					else {
						optimal = false;
						break;
					}
				}
			}
			else {
				DataSet add = null;
				for(Queue<DataSet> q : map.values()) {
					if(!q.isEmpty()) {
						add = q.poll();
						break;
					}
				}

				addRow(add,i);

			}


		}


	}


	public void addRow(DataSet d, int i) {
		
		if (i > numExamples() || d == null) {
			throw new IllegalArgumentException("Invalid index for adding a row");
		}
		
		//getFirst().putRow(i, d.getFirst());
		getFirst().assignRow( i, d.getFirst().viewRow(0) );
		
		//getSecond().putRow(i,d.getSecond());
		getSecond().assignRow( i, d.getSecond().viewRow(0) );
	}


	private int getLabel(DataSet data) {
		
		//return SimpleBlas.iamax(data.getSecond());
		
		return MatrixUtils.iamax( data.getSecond().viewRow(0) );
		
	}


	public Matrix exampleSums() {
		//return getFirst().columnSums();
		return MatrixUtils.columnSums(getFirst());
	}

/*	public Matrix exampleMaxs() {
		return getFirst().columnMaxs();
	}
*/
	public Matrix exampleMeans() {
		//return getFirst().columnMeans();
		return MatrixUtils.columnMeans(getFirst());
	}
/*
	public void saveTo(File file,boolean binary) throws IOException {
		if(file.exists())
			file.delete();
		file.createNewFile();

		if(binary) {
			DataOutputStream bos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
			getFirst().out(bos);
			getSecond().out(bos);
			bos.flush();
			bos.close();

		}
		else {
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(file));
			for(int i = 0; i < numExamples(); i++) {
				bos.write(getFirst().getRow(i).toString("%.3f", "[", "]", ", ", ";").getBytes());
				bos.write("\t".getBytes());
				bos.write(getSecond().getRow(i).toString("%.3f", "[", "]", ", ", ";").getBytes());
				bos.write("\n".getBytes())	;


			}
			bos.flush();
			bos.close();

		}
	}
*/
/*
	public static DataSet load(File path) throws IOException {
		DataInputStream bis = new DataInputStream(new BufferedInputStream(new FileInputStream(path)));
		Matrix x = new DenseMatrix(1,1);
		Matrix y = new DenseMatrix(1,1);
		x.in(bis);
		y.in(bis);
		bis.close();
		return new DataSet(x,y);
	}
*/
	/**
	 * Sample without replacement and a random rng
	 * @param numSamples the number of samples to get
	 * @return a sample data set without replacement
	 */
	public DataSet sample(int numSamples) {
		return sample(numSamples,new MersenneTwister(System.currentTimeMillis()));
	}

	/**
	 * Sample without replacement
	 * @param numSamples the number of samples to get
	 * @param rng the rng to use
	 * @return the sampled dataset without replacement
	 */
	public DataSet sample(int numSamples,RandomGenerator rng) {
		return sample(numSamples,rng,false);
	}

	/**
	 * Sample a dataset numSamples times
	 * @param numSamples the number of samples to get
	 * @param withReplacement the rng to use
	 * @return the sampled dataset without replacement
	 */
	public DataSet sample(int numSamples,boolean withReplacement) {
		return sample(numSamples,new MersenneTwister(System.currentTimeMillis()),withReplacement);
	}

	/**
	 * Sample a dataset
	 * @param numSamples the number of samples to get
	 * @param rng the rng to use
	 * @param withReplacement whether to allow duplicates (only tracked by example row number)
	 * @return the sample dataset
	 */
	public DataSet sample(int numSamples,RandomGenerator rng,boolean withReplacement) {
		if(numSamples >= numExamples())
			return this;
		else {
			Matrix examples = new DenseMatrix(numSamples,getFirst().numCols());
			Matrix outcomes = new DenseMatrix(numSamples,numOutcomes());
			Set<Integer> added = new HashSet<Integer>();
			for(int i = 0; i < numSamples; i++) {
				int picked = rng.nextInt(numExamples());
				while(added.contains(picked)) {
					picked = rng.nextInt(numExamples());

				}
				//examples.putRow(i,getFirst().getRow(i));
				examples.assignRow( i, getFirst().viewRow(i) );
				//outcomes.putRow(i,getSecond().getRow(i));
				outcomes.assignRow( i, getSecond().viewRow(i) );

			}
			return new DataSet(examples,outcomes);
		}
	}

	public void roundToTheNearest(int roundTo) {
		
		for (int i = 0; i < MatrixUtils.length( getFirst() ); i++) {
			
			//double curr = getFirst().get(i);
			double curr = MatrixUtils.getElement( getFirst(), i );
			
			//getFirst().put(i,MathUtils.roundDouble(curr, roundTo));
			MatrixUtils.setElement( getFirst(), i, curr );
			
		}
		
	}

	public int numOutcomes() {
		return getSecond().numCols();
	}

	public int numExamples() {
		return getFirst().numRows();
	}

	
	

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("===========INPUT===================\n")
		.append(getFirst().toString().replaceAll(";","\n"))
		.append("\n=================OUTPUT==================\n")
		.append(getSecond().toString().replaceAll(";","\n"));
		return builder.toString();
	}
/*
	public static void main(String[] args) throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(100);
		DataSet write = new DataSet(fetcher.next());
		write.saveTo(new File(args[0]), false);


	}
*/
}