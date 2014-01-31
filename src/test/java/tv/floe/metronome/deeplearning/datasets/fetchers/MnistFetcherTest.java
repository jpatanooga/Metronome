package tv.floe.metronome.deeplearning.datasets.fetchers;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;


public class MnistFetcherTest extends BaseDataFetcherTest {

	
	
	@Test
	public void testMnistFetcher() throws IOException {
		Pair<Matrix, Matrix> pair = DeepLearningTest.getMnistExample(1);
		int inputColumns = pair.getFirst().columns;
		int outputColumns = 10;
		testFetcher(fetcher, inputColumns, outputColumns);
		testFetchBatchSize(10);
		assertEquals(true,fetcher.hasMore());
	}
	
	@Override
	public DataSetFetcher getFetcher() {
		try {
			return new MnistDataFetcher();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	

}