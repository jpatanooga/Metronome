package tv.floe.metronome.clustering.kmeans;

import static org.junit.Assert.*;

import org.junit.Test;

import tv.floe.metronome.clustering.kmeans.Point;

public class PointTest {

	@Test
	public void testDimensionality() {
		double [] d = {3, 2};
		Point p = new Point(d);
		assertEquals(d.length, p.dimensionality());
	}
	
	@Test
	public void testGet() {
		double [] d = {3, 4, 5};
		Point p = new Point(d);
		for(int i = 0; i < d.length; i++) {
			assertEquals(d[i], p.get(i), 0);
		}
	}

	@Test
	public void testGetMinus() {
		double [] d = {3, 4, 5};
		Point p = new Point(d);
		for(int i = 0; i < d.length; i++) {
			assertEquals(d[i], p.get(i), 0);
		}
	}
}
