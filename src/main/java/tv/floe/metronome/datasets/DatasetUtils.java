package tv.floe.metronome.datasets;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;

import org.apache.commons.io.FileUtils;

public class DatasetUtils {

	
    public static File downloadAndUntar(String localDirName, String trainingFilesFilename, String trainingFilesURL ) throws IOException {
//        if(fileDir != null) {
 //         return fileDir;
   //     }
        // mac gives unique tmp each run and we want to store this persist
        // this data across restarts
        File tmpDir = new File("/tmp");
        if(!tmpDir.isDirectory()) {
          tmpDir = new File(System.getProperty("java.io.tmpdir"));
        }
        File baseDir = new File(tmpDir, localDirName);
        if(!(baseDir.isDirectory() || baseDir.mkdir())) {
          throw new IOException("Could not mkdir " + baseDir);
        }
        
        
        
        // get training records
        File tarFile = new File(baseDir, trainingFilesFilename);
        
        if(!tarFile.isFile()) {
        	System.out.println("Downloading training data....");
          FileUtils.copyURLToFile(new URL(trainingFilesURL), tarFile);      
        } else {
        	
        	System.out.println("Using cached local training data: " + trainingFilesFilename);
        	
        }
        
        System.out.println("Unzipping training data....");
        gunzipFile(baseDir, tarFile);
        
/*        File tarLabelsFile = new File(baseDir, trainingFileLabelsFilename);
        
        if(!tarLabelsFile.isFile()) {
        	System.out.println("Downloading training label data....");
          FileUtils.copyURLToFile(new URL(trainingFileLabelsURL), tarLabelsFile);      
        }

        System.out.println("Unzipping training label data....");
        gunzipFile(baseDir, tarLabelsFile);
        */
        
        
        
        
        //fileDir = baseDir;
        return baseDir;
      }
    
    public static void untarFile(File baseDir, File tarFile) throws IOException {
    	
    	System.out.println("Untaring File: " + tarFile.toString());
        
        Process p = Runtime.getRuntime().exec(String.format("tar -C %s -xvf %s", 
            baseDir.getAbsolutePath(), tarFile.getAbsolutePath()));
        BufferedReader stdError = new BufferedReader(new 
            InputStreamReader(p.getErrorStream()));
        System.out.println("Here is the standard error of the command (if any):\n");
        String s;
        while ((s = stdError.readLine()) != null) {
            System.out.println(s);
        }
        stdError.close();
    	
    	
    }

    public static void gunzipFile(File baseDir, File gzFile) throws IOException {
    	
    	System.out.println("gunzip'ing File: " + gzFile.toString());
        
        Process p = Runtime.getRuntime().exec(String.format("gunzip %s", 
            gzFile.getAbsolutePath()));
        BufferedReader stdError = new BufferedReader(new 
            InputStreamReader(p.getErrorStream()));
        System.out.println("Here is the standard error of the command (if any):\n");
        String s;
        while ((s = stdError.readLine()) != null) {
            System.out.println(s);
        }
        stdError.close();
    	
    	
    }	
	
	
}
