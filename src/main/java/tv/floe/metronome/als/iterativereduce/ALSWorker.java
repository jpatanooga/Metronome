package tv.floe.metronome.als.iterativereduce;

import java.util.List;

import net.myrrix.common.LangUtils;
import net.myrrix.common.collection.FastByIDFloatMap;
import net.myrrix.common.collection.FastByIDMap;
import net.myrrix.common.math.MatrixUtils;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Pair;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tv.floe.metronome.classification.neuralnetworks.iterativereduce.WeightsUpdateable;
import tv.floe.metronome.linearregression.iterativereduce.NodeBase;
















//
import com.cloudera.iterativereduce.ComputableWorker;
import com.cloudera.iterativereduce.io.RecordParser;

/**
 * TODO: Why is NodeBase in linearregression namespace?
 * 
 * @author josh
 *
 */
public class ALSWorker extends NodeBase implements ComputableWorker<WeightsUpdateable> {

	private static final Logger log = LoggerFactory.getLogger(ALSWorker.class);
	
	public static final double DEFAULT_ALPHA = 1.0;
	public static final double DEFAULT_LAMBDA = 0.1;
	public static final double DEFAULT_CONVERGENCE_THRESHOLD = 0.001;
	public static final int DEFAULT_MAX_ITERATIONS = 30;
	
	private static final int WORK_UNIT_SIZE = 100;
	private static final int NUM_USER_ITEMS_TO_TEST_CONVERGENCE = 100;
	
	private static final long LOG_INTERVAL = 100000;
	private static final int MAX_FAR_FROM_VECTORS = 100000;
	  // This will cause the ALS algorithm to reconstruction the input matrix R, rather than the
	  // matrix P = R > 0 . Don't use this unless you understand it!
	  private static final boolean RECONSTRUCT_R_MATRIX = 
	      Boolean.parseBoolean(System.getProperty("model.reconstructRMatrix", "false"));
	  // Causes the loss function to exclude entries for any input pairs that do not appear in the
	  // input and are implicitly 0
	  // Likewise, don't touch this for now unless you know what it does.
	  private static final boolean LOSS_IGNORES_UNSPECIFIED = 
	      Boolean.parseBoolean(System.getProperty("model.lossIgnoresUnspecified", "false"));



 	private final int features;
    private final FastByIDMap<float[]> Y;
    private final RealMatrix YTY;
    private final FastByIDMap<float[]> X;
    //private final Iterable<Pair<Long, FastByIDFloatMap>> workUnit;
    private final Iterable<Pair<Long, FastByIDFloatMap>> workUnit;

    private ALSWorker(int features,
                   FastByIDMap<float[]> Y,
                   RealMatrix YTY,
                   FastByIDMap<float[]> X,
                   Iterable<Pair<Long, FastByIDFloatMap>> workUnit) {
      this.features = features;
      this.Y = Y;
      this.YTY = YTY;
      this.X = X;
      this.workUnit = workUnit;
    }

    //public Void call() {
    /**
     * Main worker "call()" method from ALS Original
     * 
     */
    public WeightsUpdateable compute() {
      double alpha = getAlpha();
      double lambda = getLambda() * alpha;
      int features = this.features;
      // Each worker has a batch of rows to compute:
      for (Pair<Long,FastByIDFloatMap> work : workUnit) {

        // Row (column) in original R matrix containing total association value. For simplicity we will
        // talk about users and rows only in the comments and variables. It's symmetric for columns / items.
        // This is Ru:
        FastByIDFloatMap ru = work.getSecond();

        // Start computing Wu = (YT*Cu*Y + lambda*I) = (YT*Y + YT*(Cu-I)*Y + lambda*I),
        // by first starting with a copy of YT * Y. Or, a variant on YT * Y, if LOSS_IGNORES_UNSPECIFIED is set
        RealMatrix Wu = 
            LOSS_IGNORES_UNSPECIFIED ? 
            partialTransposeTimesSelf(Y, YTY.getRowDimension(), ru.keySetIterator()) : 
            YTY.copy();

        double[][] WuData = MatrixUtils.accessMatrixDataDirectly(Wu);
        double[] YTCupu = new double[features];

        for (FastByIDFloatMap.MapEntry entry : ru.entrySet()) {

          double xu = entry.getValue();

          float[] vector = Y.get(entry.getKey());
          if (vector == null) {
            log.warn("No vector for {}. This should not happen. Continuing...", entry.getKey());
            continue;
          }

          // Wu and YTCupu
          if (RECONSTRUCT_R_MATRIX) {
            for (int row = 0; row < features; row++) {
              YTCupu[row] += xu * vector[row];
            }
          } else {
            double cu = 1.0 + alpha * FastMath.abs(xu);            
            for (int row = 0; row < features; row++) {
              float vectorAtRow = vector[row];
              double rowValue = vectorAtRow * (cu - 1.0);
              double[] WuDataRow = WuData[row];              
              for (int col = 0; col < features; col++) {
                WuDataRow[col] += rowValue * vector[col];
                //Wu.addToEntry(row, col, rowValue * vector[col]);
              }
              if (xu > 0.0) {
                YTCupu[row] += vectorAtRow * cu;
              }
            }
          }

        }

        double lambdaTimesCount = lambda * ru.size();
        for (int x = 0; x < features; x++) {
          WuData[x][x] += lambdaTimesCount;          
          //Wu.addToEntry(x, x, lambdaTimesCount);
        }

        float[] xu = MatrixUtils.getSolver(Wu).solveDToF(YTCupu);

        // Store result:
        synchronized (X) {
          X.put(work.getFirst(), xu);
        }

        // Process is identical for computing Y from X. Swap X in for Y, Y for X, i for u, etc.
      }
      
      return null;
    }

    private static double getAlpha() {
      String alphaProperty = System.getProperty("model.als.alpha");
      return alphaProperty == null ? DEFAULT_ALPHA : LangUtils.parseDouble(alphaProperty);
    }

    private static double getLambda() {
      String lambdaProperty = System.getProperty("model.als.lambda");
      return lambdaProperty == null ? DEFAULT_LAMBDA : LangUtils.parseDouble(lambdaProperty);
    }

    /**
     * Like {@link MatrixUtils#transposeTimesSelf(FastByIDMap)}, but instead of computing MT * M, 
     * it computes MT * C * M, where C is a diagonal matrix of 1s and 0s. This is like pretending some
     * rows of M are 0.
     * 
     * @see MatrixUtils#transposeTimesSelf(FastByIDMap) 
     * @see #LOSS_IGNORES_UNSPECIFIED
     */
    private static RealMatrix partialTransposeTimesSelf(FastByIDMap<float[]> M, 
                                                        int dimension, 
                                                        LongPrimitiveIterator keys) {
      RealMatrix result = new Array2DRowRealMatrix(dimension, dimension);
      while (keys.hasNext()) {
        long key = keys.next();
        float[] vector = M.get(key);
        for (int row = 0; row < dimension; row++) {
          float rowValue = vector[row];
          for (int col = 0; col < dimension; col++) {
            result.addToEntry(row, col, rowValue * vector[col]);
          }
        }
      }
      return result;
    }
	
    @Override
	public boolean IncrementIteration() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public WeightsUpdateable compute(List<WeightsUpdateable> arg0) {
		return this.compute();
	}
	
	@Override
	public WeightsUpdateable getResults() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public void setRecordParser(RecordParser arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void setup(Configuration arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void update(WeightsUpdateable arg0) {
		// TODO Auto-generated method stub
		
	}	
	
}
