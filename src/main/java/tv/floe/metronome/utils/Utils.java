package tv.floe.metronome.utils;

/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.zip.GZIPInputStream;

import org.apache.commons.compress.archivers.ArchiveException;
import org.apache.commons.compress.archivers.ArchiveStreamFactory;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.utils.IOUtils;
import org.apache.mahout.math.Vector;

public class Utils {

  
  public static void PrintVector(Vector v) {
    
    boolean first = true;
    Iterator<Vector.Element> nonZeros = v.iterator();
    while (nonZeros.hasNext()) {
      Vector.Element vec_loc = nonZeros.next();
      
      if (!first) {
        System.out.print(",");
      } else {
        first = false;
      }
      
      System.out.print(" " + vec_loc.get());
      
    }
    
    System.out.println("");
    
  }
  
  public static void PrintVectorSection(Vector v, int num) {
    
    boolean first = true;
    Iterator<Vector.Element> nonZeros = v.iterator();
    int cnt = 0;
    
    while (nonZeros.hasNext()) {
      Vector.Element vec_loc = nonZeros.next();
      
      if (!first) {
        System.out.print(",");
      } else {
        first = false;
      }
      
      System.out.print(" " + vec_loc.get());
      if (cnt > num) {
        break;
      }
      cnt++;
    }
    
    System.out.println(" ######## ");
    
  }
  
  public static void PrintVectorNonZero(Vector v) {
    
    boolean first = true;
    Iterator<Vector.Element> nonZeros = v.iterateNonZero();
    while (nonZeros.hasNext()) {
      Vector.Element vec_loc = nonZeros.next();
      
      if (!first) {
        System.out.print(",");
      } else {
        first = false;
      }
      System.out.print(" " + vec_loc.get());
      
    }
    
    System.out.println("");
    
  }
  
  public static void PrintVectorSectionNonZero(Vector v, int size) {
    
    boolean first = true;
    Iterator<Vector.Element> nonZeros = v.iterateNonZero();
    
    int cnt = 0;
    
    while (nonZeros.hasNext()) {
      Vector.Element vec_loc = nonZeros.next();
      
      if (!first) {
        System.out.print(",");
      } else {
        first = false;
      }
      System.out.print(" " + vec_loc.get());
      
      if (cnt > size) {
        break;
      }
      cnt++;
    }
    
    System.out.println("");
    
  }
  
  
}
