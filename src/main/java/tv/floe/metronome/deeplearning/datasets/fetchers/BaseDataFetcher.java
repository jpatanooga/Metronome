package tv.floe.metronome.deeplearning.datasets.fetchers;

import java.util.List;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tv.floe.metronome.deeplearning.datasets.DataSet;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;



public abstract class BaseDataFetcher implements DataSetFetcher {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -859588773699432365L;
	protected int cursor = 0;
	protected int numOutcomes = -1;
	protected int inputColumns = -1;
	protected DataSet curr;
	protected int totalExamples;
	protected static Logger log = LoggerFactory.getLogger(BaseDataFetcher.class);
	
	
	protected Matrix createInputMatrix(int numRows) {
		return new DenseMatrix(numRows,inputColumns);
	}
	
	protected Matrix createOutputVector(int outcomeLabel) {
		return MatrixUtils.toOutcomeVector(outcomeLabel, numOutcomes);
	}
	
	protected Matrix createOutputMatrix(int numRows) {
		return new DenseMatrix(numRows,numOutcomes);
	}
	
	protected void initializeCurrFromList(List<Pair<Matrix,Matrix>> examples) {
		
		if(examples.isEmpty())
			log.warn("Warning: empty dataset from the fetcher");
		
		Matrix inputs = createInputMatrix(examples.size());
		Matrix labels = createOutputMatrix(examples.size());
		
		for (int i = 0; i < examples.size(); i++) {
		
			//inputs.putRow(i, examples.get(i).getFirst());
			inputs.assignRow( i, examples.get(i).getFirst().viewRow(0) );
			//labels.putRow(i,examples.get(i).getSecond());
			labels.assignRow( i, examples.get(i).getSecond().viewRow(0) );
		
		}
		
		curr = new DataSet(inputs,labels);

	}
	
	@Override
	public boolean hasMore() {
		return cursor < totalExamples;
	}

	@Override
	public DataSet next() {
		return curr;
	}

	@Override
	public int totalOutcomes() {
		return numOutcomes;
	}

	@Override
	public int inputColumns() {
		return inputColumns;
	}

	@Override
	public int totalExamples() {
		return totalExamples;
	}

	@Override
	public void reset() {
		cursor = 0;
	}

	@Override
	public int cursor() {
		return cursor;
	}
	
	

	
}