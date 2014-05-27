package tv.floe.metronome.deeplearning.dbn.model.evaluation;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import java.util.Properties;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.cloudera.iterativereduce.ConfigFields;


public class YarnCmdLineModelTester extends Configured implements Tool {

	private static final Log LOG = LogFactory.getLog(YarnCmdLineModelTester.class);

	@Override
	public int run(String[] args) throws Exception {

		// System.out.println("IR: Client.run() [start]");

		if (args.length < 1)
			LOG.info("No configuration file specified, using default ("
					+ ConfigFields.DEFAULT_CONFIG_FILE + ")");

		long startTime = System.currentTimeMillis();
		String configFile = (args.length < 1) ? ConfigFields.DEFAULT_CONFIG_FILE
				: args[0];
		Properties props = new Properties();
		Configuration conf = getConf();
		
		// TODO: let them pass this in
		// let's assume we're not going through a hadoop/yarn bash script
		conf.addResource( new Path("/etc/hadoop/conf/core-site.xml") );

		try {
			FileInputStream fis = new FileInputStream(configFile);
			props.load(fis);
		} catch (FileNotFoundException ex) {
			throw ex; // TODO: be nice
		} catch (IOException ex) {
			throw ex; // TODO: be nice
		}

		// Make sure we have some bare minimums
		ConfigFields.validateConfig(props);

		if (LOG.isDebugEnabled()) {
			LOG.debug("Loaded configuration: ");
			for (Map.Entry<Object, Object> entry : props.entrySet()) {
				LOG.debug(entry.getKey() + "=" + entry.getValue());
			}
		}

		// TODO: make sure input file(s), libs, etc. actually exist!
		// Ensure our input path exists

		Path p = new Path(props.getProperty(ConfigFields.APP_INPUT_PATH));
		FileSystem fs = FileSystem.get(conf);

		if (!fs.exists(p)) {
			throw new FileNotFoundException("Input path not found: "
					+ p.toString() + " (in " + fs.getUri() + ")");
		} else {
			
			//System.out.println( "Using FS at: " + conf.get("fs.defaultFS") );
			
		}

		LOG.info("Using input path: " + p.toString());
		
		
		
		ModelTester.evaluateModel( configFile, conf, 20 );
		
		
		
		return 0;



	}

	public static void main(String[] args) throws Exception {
		int rc = ToolRunner.run(new Configuration(), new YarnCmdLineModelTester(), args);

		// Log, because been bitten before on daemon threads; sanity check
		LOG.debug("Calling System.exit(" + rc + ")");
		System.exit(rc);
	}
}