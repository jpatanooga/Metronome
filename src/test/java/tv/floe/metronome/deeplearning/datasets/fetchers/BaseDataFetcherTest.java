package tv.floe.metronome.deeplearning.datasets.fetchers;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeNotNull;

import org.junit.Before;
import org.junit.Test;

public abstract class BaseDataFetcherTest {

	protected DataSetFetcher fetcher;
	
	@Before
	public void setup() {
		fetcher = getFetcher();
		assumeNotNull(fetcher);
	}
	
	
	public void testFetcher(DataSetFetcher fetcher,int inputColumnsExpected,int totalOutcomesExpected) {
		assertEquals(inputColumnsExpected,fetcher.inputColumns());
		assertEquals(totalOutcomesExpected,fetcher.totalOutcomes());
		assertEquals(true,fetcher.hasMore());
		assertEquals(true,fetcher.next() == null && fetcher.hasMore());
		
	}
	
	public void testFetchBatchSize(int expectedBatchSize) {
		fetcher.fetch(expectedBatchSize);
		assumeNotNull(fetcher.next());
		assertEquals(expectedBatchSize,fetcher.next().getFirst().numRows());
		
	}
	

	
	
	
	public abstract DataSetFetcher getFetcher();

}