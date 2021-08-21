package txtmine;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class MatrixGenerator {
	public MatrixGenerator(){
	}
	
	public void genreg(HashMap<String, Integer> words, Map<String, Integer> doc, int[][] mtx, int row) {
		int [][] arr = mtx;
		for(Entry<String, Integer> entry : doc.entrySet()) {
			String key = entry.getKey();
			Integer value = entry.getValue();
			int wordcol = words.get(key);
			arr[row][wordcol] = value;
		}
	}
	
    public double[][] tfidf(int [][] arr, int row, int col) {
    	double [][] tfidf = new double[row][col];
    	int [] tf = new int[row];
		int [] idf = new int[col];
		
		for(int i = 0; i < row; i++) {
			int totdoc = 0;
			for(int j = 0; j< col; j++) {
				totdoc += arr[i][j];
			}
			tf[i] = totdoc;
		}
		
		for(int i = 0; i < col; i++) {
			int numtim = 0;
			for(int j = 0; j < row; j++) {
				if(arr[j][i] > 0) {
					numtim++;
				}
			}
			idf[i] = numtim;
		}
		
		for(int i = 0; i < row; i++) {
			for(int j = 0; j < col; j++) {
				double temptf = arr[i][j];
				double temptf2 = tf[i];
				double temptf3 = temptf/temptf2;
				double tempidf = row/idf[j];
				tempidf = Math.log(tempidf);
				tfidf[i][j] = temptf3 * tempidf;
			}
		}
		
    	return tfidf;
    }
    
    public HashMap<String, Double> clusTxt(double [][] arr, HashMap<Integer, String> fcol, int start, int end){
    	HashMap<String, Double> sumM = new HashMap<String, Double>();
		for(int i = 0; i < fcol.size(); i++) {
			double totnum = 0;
			for(int j = start; j< end; j++) {
				totnum += arr[j][i];
			}
			sumM.put(fcol.get(i), totnum);
		}
    	return sumM;
    }
    
    public int [] convt(String [] sarr) {
    	int []  arr = new int [sarr.length];
    	for(int i = 0; i < sarr.length; i++) {
    		if(sarr[i] == "c1") {
    			arr[i] = 0;
    		}
    		else if(sarr[i] == "c4") {
    			arr[i] = 1;
    		}
    		else if(sarr[i] == "c7") {
    			arr[i] = 2;
    		}
    	}
    	return arr;
    }
    
    public int [][] confMtx(int [] assign, int [] actualAssign){
    	int [][]  arr = new int [3][3];
    	for(int i = 0; i < assign.length; i++) {
    		int row = assign[i];
    		int col = actualAssign[i];
    		arr[row][col] = arr[row][col] + 1;
    	}
    	return arr;
    }
    
    public String printAnlys(int[][] confMtx, int clus) {
    	int adjClus = clus - 1;
    	double tp = confMtx[adjClus][adjClus];
    	double fp = 0;
    	double fn = 0;
    	for(int i = 0; i < confMtx.length; i++) {
    		fp += confMtx[adjClus][i];
    	}
    	for(int i = 0; i < confMtx.length; i++) {
    		fn += confMtx[i][adjClus];
    	}
    	fp = fp - tp;
    	fn = fn - tp;
    	
    	double precision = tp/(tp+fp);
    	double recall = tp/(tp+fn);
    	double fscore = 2 * (precision * recall)/(precision + recall);
    	return "Precision: " + String.format("%.3f",precision) + " " + "Recall: " + String.format("%.3f",recall) + " " + "F-1_Score: " + String.format("%.3f",fscore);
    }
}
