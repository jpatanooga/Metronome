package tv.floe.metronome.io;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class FileUtils {

	public static OutputStream createAppendingOutputStream(File to) {
		try {
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(to,true));
			return bos;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
}
