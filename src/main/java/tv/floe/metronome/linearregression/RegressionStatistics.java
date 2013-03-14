package tv.floe.metronome.linearregression;

/**
 * Base mechanics to measure R-squared for a regression
 * 
 * @author josh
 *
 */
public class RegressionStatistics {
	
	private double y_total = 0;
	private long record_count = 0;
	
	// these summed values should have been squared before being partially summed
	private double total_sum_SSyy = 0;

	// these summed values should have been squared before being partially summed	
	private double total_sum_SSE = 0;
	
	
	public double CalculateRSquared() {
		
		return 1 - ( this.GetSSE() / this.GetSSyy() );
	}
	
	public void AddPartialSumForY(double partial_y_sum, long count) {
		
		this.y_total += partial_y_sum;
		this.record_count += count;
		
	}
	
	/**
	 * This is a component used in the calculation of SSyy
	 * 
	 * @return
	 */
	public double ComputeYAvg() {
		
		return this.y_total / this.record_count;
		
	}
	
	/**
	 * "Sum of Squares of Model"
	 * 
	 * measures the deviations of the observations (y) from their mean (y-bar)
	 * 
	 * - its only really a "partial" sum until the last partial sum is accumulated
	 * 
	 * @return
	 */
	public double GetSSyy() {
		
		return this.total_sum_SSyy;
		
	}
	
	/**
	 * Accumulates the squared differences between observed-y and the observed-y mean
	 * 
	 * @param partial_SSyy_sum
	 */
	public void AccumulateSSyyPartialSum(double partial_SSyy_sum) {
		this.total_sum_SSyy += partial_SSyy_sum;
	}
	
	/**
	 * "Sum of Squares of Error"
	 * 
	 * measures the deviations of observations from their predicted values
	 * 
	 * - its only really a "partial" sum until the last partial sum is accumulated
	 * 
	 * @return
	 */
	public double GetSSE() {
		
		return this.total_sum_SSE;
		
	}
	
	public void AccumulateSSEPartialSum(double partial_SSE_sum) {
		
		this.total_sum_SSE += partial_SSE_sum;
		
	}
	
	
	

}
