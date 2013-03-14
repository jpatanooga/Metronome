package tv.floe.metronome.linearregression;

public class QuickTestConfig {

	public String fileName = "";
	public double learningRate = 0;
	public int iterations = 1;
	public int feature_size = 2;
	
	public QuickTestConfig( String file_name, double lr, int iterations, int featureSize ) {
		this.fileName = file_name;
		this.learningRate = lr;
		this.iterations = iterations;
		this.feature_size = featureSize;
	}
	
}
