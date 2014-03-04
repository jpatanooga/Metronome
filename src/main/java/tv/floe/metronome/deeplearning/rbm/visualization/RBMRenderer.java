package tv.floe.metronome.deeplearning.rbm.visualization;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.apache.mahout.math.Matrix;

import tv.floe.metronome.deeplearning.rbm.RestrictedBoltzmannMachine;
import tv.floe.metronome.math.MatrixUtils;
import tv.floe.metronome.types.Pair;

public class RBMRenderer {

	

	public  JFrame frame;
	BufferedImage img;
	private int width = 28;
	private int height = 28;
	public String title = "TEST";
	private int heightOffset = 0;
	private int widthOffset = 0;

	/*
	public RBMRenderer(Matrix data,int heightOffset,int widthOffset) {
		
		img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		this.heightOffset = heightOffset;
		this.widthOffset = widthOffset;
		WritableRaster r = img.getRaster();
		int[] equiv = new int[ MatrixUtils.length( data ) ];
		
		for (int i = 0; i < equiv.length; i++) {
			
			equiv[i] = (int) Math.round( MatrixUtils.getElement(data, i) );
			
		}
		
		r.setDataElements(0, 0, width, height, equiv);


	}
	
	public RBMRenderer(Matrix data) {
		
		img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		WritableRaster r = img.getRaster();
		int[] equiv = new int[ MatrixUtils.length( data ) ];
		
		for (int i = 0; i < equiv.length; i++) {
			
			equiv[i] = (int) Math.round( MatrixUtils.getElement(data, i) );
			
		}
		
		r.setDataElements(0, 0, width, height, equiv);


	}
	*/
	
	public RBMRenderer() { }
	
	public void renderHiddenBiases(int heightOffset, int widthOffset, Matrix render_data, String filename) {
		/* -- python code
		 * 
    hMean = from_file(path)
    image = Image.fromarray(hMean * 256).show()

		 * 
		 */
		
		this.width = render_data.numCols();
		this.height = render_data.numRows();
		
		img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		this.heightOffset = heightOffset;
		this.widthOffset = widthOffset;
		WritableRaster r = img.getRaster();
		int[] equiv = new int[ MatrixUtils.length( render_data ) ];
		
		for (int i = 0; i < equiv.length; i++) {
			
			//equiv[i] = (int) Math.round( MatrixUtils.getElement(render_data, i) );
			equiv[i] = (int) Math.round( MatrixUtils.getElement(render_data, i) * 256 );
			System.out.println( "> " + equiv[i] );
			
		}
		
		System.out.println( "hbias size: Cols: " + render_data.numCols() + ", Rows: " + render_data.numRows()  );
		
		r.setDataElements(0, 0, width, height, equiv);
		
		//d.saveToDisk("/tmp/Metronome/RBM/" + UUIDForRun + "/" + number + "/" + number + "_real.png");
		this.saveToDisk(filename);
		
	}
	
	/**
	 * 
	 * Figure 2. Histograms of hBias, W, vBias (top row) 
	 * and the last batch updates to each (bottom row). 
	 * 
	 */
	public void renderAllHistograms(RestrictedBoltzmannMachine rbm) {
		
		
		
	}
	
	public int computeHistogramBucketIndex(double min, double stepSize, double value, int numberBins) {
		
		for ( int x = 0; x < numberBins; x++ ) {
			
			double tmp = (x * stepSize) + min;
			
			if ( value >= tmp && value <= (tmp + stepSize) ) {
				return x;
			}
			
		}
		
		return -10;
		
	}
	
	/**
	 * 
	 * This is faster but produces rounding errors
	 * 
	 * @param min
	 * @param stepSize
	 * @param value
	 * @param numberBins
	 * @return
	 */
	public int computeHistogramBucketIndexAlt(double min, double stepSize, double value, int numberBins) {
		
		
	//	System.out.println("pre round: val: " + value + ", delta on min: " + (value - min) + ", bin-calc: " + ((value - min) / stepSize));
		System.out.println("pre round: val: " + value + ", bin-calc: " + ((value - min) / stepSize));
		
		
		
		// int bin = (int) ((value - min) / stepSize);
		
		int bin = (int) (((value - min)) / (stepSize));
		
		/*
		for ( int x = 0; x < numberBins; x++ ) {
			
			double tmp = (x * stepSize) + min;
			
			if ( value <= tmp ) {
				return x;
			}
			
		}
		*/
		return bin;
		
	}	
	
	private String buildBucketLabel(int bucketIndex, double stepSize, double min) {
		
		double val = min + (bucketIndex * stepSize);
		String ret = "" + val;
		
		return ret;
				
	}
	
	/**
	 * Take some matrix input data and a bucket count and compute:
	 * 
	 * - a list of N buckets, each with:
	 * 1. a bucket label
	 * 2. a bucket count
	 * 
	 * over the input dataset
	 * 
	 * @param data
	 * @param numberBins
	 * @return
	 */
	//public Map<Integer, Pair<String, Integer>> generateHistogramBuckets(Matrix data, int numberBins) {
	public Map<Integer, Integer> generateHistogramBuckets(Matrix data, int numberBins) {
		
		//Pair<> p = new Pair<>();
//		Map<Integer, Pair<String, Integer>> mapHistory = new TreeMap<Integer, Pair<String, Integer>>(); 
		Map<Integer, Integer> mapHistory = new TreeMap<Integer, Integer>();
		
		//int binCount = 10;
		double min = MatrixUtils.min(data); //data.get(0, 0);
		double max = MatrixUtils.max(data); //data.get(0, 0);
		
		double range = max - min;
		double stepSize = range / numberBins;
		
		System.out.println( "min: " + min );
		System.out.println( "max: " + max );
		System.out.println( "range: " + range );
		System.out.println( "stepSize: " + stepSize );
		System.out.println( "numberBins: " + numberBins );
		
		//stepSize = 1;
		
		for ( int row = 0; row < data.numRows(); row++ ) {
			
			for (int col = 0; col < data.numCols(); col++ ) {
		 	
				double matrix_value = data.get( row, col );
				
		 		// at this point we need round values into bins
				
				int bucket_key = this.computeHistogramBucketIndex(min, stepSize, matrix_value, numberBins);
				
			//	int bucket_key_alt = this.computeHistogramBucketIndexAlt(min, stepSize, matrix_value, numberBins);

			//	System.out.println("> bin key: " + bucket_key + ", alt: " + bucket_key_alt);
				
                //int amount = 0;
				//Pair<String, Integer> entry = null;
				int entry = 0;
				
                if (mapHistory.containsKey( bucket_key )) {
                	
                	// entry exists, increment
                	
                    entry = mapHistory.get( bucket_key );
                    //amount++;
                    //int tmp = entry.getSecond(); //. = entry.getSecond() + 1;
                    //tmp++;
                    entry++;
                    
                    mapHistory.put( bucket_key, entry );
                    
                } else {
                	
                	// entry does not exit, create, insert
                    //amount = 1;
                	
                	// create new key
                	//bucket_key = 1;
                	
                	// bucket label
                	String bucket_label = buildBucketLabel(bucket_key, stepSize, min);
                	
                	// new entry
                	entry = 1; // new Pair<String, Integer>(bucket_label, 1);
                
                	// update data structure
                	mapHistory.put( bucket_key, entry );
                }
                
                
                
			}
			
		}	
		
		
		
		return mapHistory;
	}
	
	
	/**
	 * Groups values into 1 of 10 bins, sums, and renders
	 * 
	 * @param data
	 * @param numberBins
	 */
	public void renderHistogram(Matrix data, int numberBins) {
		
		// TODO: how are double mapped into bins?
		// TODO: calc max
		// TODO: calc bins - we want 10 bins
		// 
		
		
		// calc bins
		
		// Map< bin-ID, count >
		Map<Integer, Integer> mapHistory = new TreeMap<Integer, Integer>();
		
		for ( int row = 0; row < data.numRows(); row++ ) {
			
			for (int col = 0; col < data.numCols(); col++ ) {
		 	
				double value = data.get( row, col );
				
		 			// at this point we need round values into bins
		 			/*
	                int amount = 0;
	                if (mapHistory.containsKey(value)) {
	                    amount = mapHistory.get(value);
	                    amount++;
	                } else {
	                    amount = 1;
	                }
	                mapHistory.put(value, amount);
	                */
			}
			
		}		
		
		
	}
	
	/**
	 * 
	 * Once the probability image and weight histograms are 
	 * behaving satisfactorily, we plot the learned filter 
	 * for each hidden neuron, one per column of W. Each filter 
	 * is of the same dimension as the input data, and it is 
	 * most useful to visualize the filters in the same way 
	 * as the input data is visualized.
	 * 
	 */
	public void renderFilters() {
		
		// for each hidden neuron
		
			// plot the learned filter (same dim as the input data)
		
		
	}
	
	

	public void renderActivations(int heightOffset, int widthOffset, Matrix activation_data, String filename, int scale ) {
		
		this.width = activation_data.numCols();
		this.height = activation_data.numRows();
		
		
		System.out.println( "----- renderActivations ------" );
		
		img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
		this.heightOffset = heightOffset;
		this.widthOffset = widthOffset;
		WritableRaster r = img.getRaster();
		int[] equiv = new int[ MatrixUtils.length( activation_data ) ];
		
		double max = 0.1 * scale; //MatrixUtils.max(render_data);
		double min = -0.1 * scale; //MatrixUtils.min(render_data);
		double range = max - min;

		
		for (int i = 0; i < equiv.length; i++) {
			
			equiv[i] = (int) Math.round( MatrixUtils.getElement(activation_data, i) * 255 );
			
		}
		
		
		System.out.println( "activations size: Cols: " + activation_data.numCols() + ", Rows: " + activation_data.numRows()  );
		
		r.setPixels(0, 0, width, height, equiv);
		
		this.saveToDisk(filename);
		
	}	
	
	
	public void renderHistogramMatrix() {
		
		
	}
	
	public void renderFilter() {
		
		
	}
	
	
	public static void saveImageToDisk(BufferedImage img, String imageName) throws IOException {
		
		File outputfile = new File( imageName );
		
		outputfile.getParentFile().mkdirs();
		//FileWriter writer = new FileWriter(file);
		
		
		ImageIO.write(img, "png", outputfile);		
		
	}
	
	public void saveToDisk(String filename) {
		
		try {
			saveImageToDisk( this.img, filename );
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	
	public void draw() {
		frame = new JFrame(title);
		frame.setVisible(true);
		start();
		frame.add(new JLabel(new ImageIcon(getImage())));

		frame.pack();
		// Better to DISPOSE than EXIT
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	}

	public void close() {
		frame.dispose();
	}
	
	public Image getImage() {
		return img;
	}

	public void start(){


		int[] pixels = ((DataBufferInt)img.getRaster().getDataBuffer()).getData();
		boolean running = true;
		while(running){
			BufferStrategy bs = frame.getBufferStrategy();
			if(bs==null){
				frame.createBufferStrategy(4);
				return;
			}
			for (int i = 0; i < width * height; i++)
				pixels[i] = 0;

			Graphics g= bs.getDrawGraphics();
			g.drawImage(img, heightOffset, widthOffset, width, height, null);
			g.dispose();
			bs.show();

		}
	}	
	
	
}
