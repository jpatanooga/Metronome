package tv.floe.metronome.math;

public class MathUtils {

	/**
	 * Normalize a value
	 * (val - min) / (max - min)
	 * @param val value to normalize
	 * @param max max value
	 * @param min min value
	 * @return the normalized value
	 */
	public static double normalize(double val,double min,double max) {
		if(max < min)
			throw new IllegalArgumentException("Max must be greather than min");
		
		return (val - min) / (max - min);
	}

}
