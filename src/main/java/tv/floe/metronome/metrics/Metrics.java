package tv.floe.metronome.metrics;

public class Metrics {

	  public long AvgBatchProecssingTimeInMS = 0;
	  
	  public long TotalInputProcessingTimeInMS = 0;
	  public long TotalRecordsProcessed = 0;
	  
	  public double AvgError = 0.0;
	  
	  double step = 0.0;
	  int[] bumps = new int[] {1, 2, 5};

	  
	  public void UpdateOnlineAvgCorrect() {
		  
	  }
	  
	  public void printProgressiveStepDebugMsg(int count, String msg) {
		  
		  int bump = bumps[(int) Math.floor(step) % bumps.length];
	        int scale = (int) Math.pow(10, Math.floor(step / bumps.length));
	        
	        if (count % (bump * scale) == 0) {
 	          //step += 0.25;
	        	step += 0.50;
	          
	          System.out.println(msg);
	          
	        }		  
		  
	  }

}
