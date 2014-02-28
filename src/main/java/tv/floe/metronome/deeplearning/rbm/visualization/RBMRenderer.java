package tv.floe.metronome.deeplearning.rbm.visualization;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.apache.mahout.math.Matrix;

import tv.floe.metronome.math.MatrixUtils;

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
	
	

	public void renderActivations(int heightOffset, int widthOffset, Matrix activation_data, String filename, int scale ) {
		
		this.width = activation_data.numCols();
		this.height = activation_data.numRows();
		
//		BufferedImage image = new BufferedImage(width, height,  
//			    BufferedImage.TYPE_BYTE_GRAY);
		
		System.out.println( "----- renderActivations ------" );
		
		img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
		this.heightOffset = heightOffset;
		this.widthOffset = widthOffset;
		WritableRaster r = img.getRaster();
		int[] equiv = new int[ MatrixUtils.length( activation_data ) ];
		
		double max = 0.1 * scale; //MatrixUtils.max(render_data);
		double min = -0.1 * scale; //MatrixUtils.min(render_data);
		double range = max - min;
		double mid_of_range = range / 2;
		/*
		System.out.println( "projected Max: " + max );
		System.out.println( "projected Min: " + min );
		System.out.println( "Range: " + range );
		System.out.println( "MiddleRange: " + mid_of_range );
		

		System.out.println( "\norig activation_data data: " );
		//MatrixUtils.debug_print( render_data );
		MatrixUtils.debug_print_row(activation_data, 0);
		*/
/*
		double normalized_max = MatrixUtils.max( activation_data );
		double normalized_min = MatrixUtils.min( activation_data );
		
		System.out.println( "actual Max: " + normalized_max );
		System.out.println( "actual Min: " + normalized_min );
	*/
		
//		Matrix normalized_render_data = activation_data.plus( mid_of_range ).divide(range);
		
		
		
		//System.out.println( "\nnormalized render data: " );
		//MatrixUtils.debug_print( normalized_render_data );
		/*
		normalized_max = MatrixUtils.max( normalized_render_data );
		normalized_min = MatrixUtils.min( normalized_render_data );
		
		System.out.println( "Max: " + normalized_max );
		System.out.println( "Min: " + normalized_min );
		*/
		
		for (int i = 0; i < equiv.length; i++) {
			
			//equiv[i] = (int) Math.round( MatrixUtils.getElement(render_data, i) );
			equiv[i] = (int) Math.round( MatrixUtils.getElement(activation_data, i) * 255 );
			//if ( i > 50 ) {
			//	equiv[i] = i;
			//} else {
				//equiv[i] = 0;
			//}
			//equiv[i] = i;
			//int tempInt = equiv[i];
			//equiv[i] = ( ( tempInt << 24 ) | ( tempInt << 16 ) | tempInt ) ; 
		//	System.out.println( "> " + equiv[i] );
			
		}
		
		//MatrixUtils.debug_print(render_data);
		
		System.out.println( "activations size: Cols: " + activation_data.numCols() + ", Rows: " + activation_data.numRows()  );
		
		//r.setDataElements(0, 0, width, height, equiv);
		r.setPixels(0, 0, width, height, equiv);
		
		//d.saveToDisk("/tmp/Metronome/RBM/" + UUIDForRun + "/" + number + "/" + number + "_real.png");
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
