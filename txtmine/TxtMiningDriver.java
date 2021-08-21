package txtmine;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class TxtMiningDriver {
    public static void main(String[] args) {
        System.out.println("Starting Preprocessing");
        
        List<String> paths = new ArrayList<String>();
        List<List> docs = new ArrayList<List>();
        HashMap<String, Integer> ngrams = new HashMap<String, Integer>();
        HashMap<String, Integer> fcol = new HashMap<String, Integer>();
        ArrayList<Map> mapd = new ArrayList<Map>();
        Preprocessing proc = new Preprocessing();
        HashMap<Integer, String> doclabels = new HashMap<Integer, String>();
        
        // Load raw data
        Collections.addAll(paths,"c1article01.txt","c1article02.txt","c1article03.txt","c1article04.txt","c1article05.txt","c1article06.txt","c1article07.txt","c1article08.txt","c4article01.txt","c4article02.txt","c4article03.txt","c4article04.txt","c4article05.txt","c4article06.txt","c4article07.txt","c4article08.txt","c7article01.txt","c7article02.txt","c7article03.txt","c7article04.txt","c7article05.txt","c7article06.txt","c7article07.txt","c7article08.txt","unknown01.txt","unknown02.txt","unknown03.txt","unknown04.txt","unknown05.txt","unknown06.txt","unknown07.txt","unknown08.txt","unknown09.txt","unknown10.txt");
        int numdocs = 34;
        
        for(int i = 0; i < 24; i++) {
        	if(i < 8) {
        		String val = "c1";
        		doclabels.put(i, val);
        	}
        	if(i > 7 && i < 16) {
        		String val = "c4";
        		doclabels.put(i, val);
        	}
        	if(i > 15 && i < 24) {
        		String val = "c7";
        		doclabels.put(i, val);
        	}
        }

        for(int i = 0; i < numdocs; i++) {
        	docs.add(proc.process(paths.get(i)));
        	proc.ngrams(ngrams, docs.get(i));
        }
	    
        // Get the iterator over the HashMap
        Iterator<Map.Entry<String, Integer> >
            iterator = ngrams.entrySet().iterator();
        // Iterate over the HashMap
        while (iterator.hasNext()) {
            // Get the entry at this iteration
            Map.Entry<String, Integer>  entry = iterator.next();
            // Check if this value is the required value  (determines how many times n-gram needs to be found)
            if (entry.getValue() < 2) {
                // Remove this entry from HashMap
                iterator.remove();
            }
        }
        
        for(int i = 0; i < numdocs; i++) {
        	 HashMap<String, Integer> temp = proc.adjForNgrams(ngrams, docs.get(i));
        	 mapd.add(temp);
        }
        
        
        fcol =  proc.adjFcol(mapd);
        HashMap<Integer, String> revfcol = proc.revFcol(fcol);
        
        MatrixGenerator mg = new MatrixGenerator();
        int [][] regmtx = new int [numdocs][fcol.size()];
        for(int i = 0; i < numdocs; i++) {
        	mg.genreg(fcol, mapd.get(i),regmtx, i);
        }
        
        // Generated tf-idf matrix for model data
        double [][] tfidf = mg.tfidf(regmtx, numdocs, fcol.size());
        
        System.out.println("----------------------------------------");
        
        Knn classify = new Knn(tfidf);
        String [] assignStr = new String [10];
        for(int i = 24; i < 34; i++) {
        	// Inputs for knn
        	classify.rknn(3, 24, true, i, fcol.size(), doclabels, assignStr);
        	int ind = i - 23;
        	String indStr = String.format("%02d", ind);
        	int len = doclabels.get(i).length() - 1;
        	System.out.println("unknown" + indStr + ".txt" + " is " + doclabels.get(i).substring(1, len));
        }
        
        String [] actualAssignStr = {"c1","c1","c1","c1","c4","c4","c7","c7","c1","c4"};
        int [] assign = mg.convt(assignStr);
        int [] actualAssign = mg.convt(actualAssignStr);
        int [][] confMtx = mg.confMtx(assign, actualAssign);
        
        System.out.println();
	    System.out.println("Confusion Matrix for Folder C1, C4, C7 (Rows- Predicted Class, Columns- Actual Class )");
	    for (int i = 0; i < 3; i++) {
	    	for (int j = 0; j < 3; j++) {
	    		System.out.printf("%3d", confMtx[i][j]);
	    	}
	    	System.out.println();
	    }
	    System.out.println();
	    
	    System.out.println("Folder C1- " + mg.printAnlys(confMtx, 1));
	    System.out.println("Folder C4- " + mg.printAnlys(confMtx, 2));
	    System.out.println("Folder C7- " + mg.printAnlys(confMtx, 3));
	    System.out.println("----------------------------------------");
	    System.out.println();
     }
}
