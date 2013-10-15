package tv.floe.metronome.als.iterativereduce;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;

public class ALSWorker {

//	private final int features;
//    private final FastByIDMap<float[]> Y;
//    private final RealMatrix YTY;
//    private final FastByIDMap<float[]> X;
//    private final Iterable<Pair<Long, FastByIDFloatMap>> workUnit;
/*
    private Worker(int features,
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
*/
    public Void call() {
//      double alpha = getAlpha();
//      double lambda = getLambda() * alpha;
 //     int features = this.features;
 /*     // Each worker has a batch of rows to compute:
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
      */
      return null;
    }

//    private static double getAlpha() {
  //    String alphaProperty = System.getProperty("model.als.alpha");
  //    return alphaProperty == null ? DEFAULT_ALPHA : LangUtils.parseDouble(alphaProperty);
//    }

//    private static double getLambda() {
      String lambdaProperty = System.getProperty("model.als.lambda");
//      return lambdaProperty == null ? DEFAULT_LAMBDA : LangUtils.parseDouble(lambdaProperty);
//    }

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
	
}
