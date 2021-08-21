package txtmine;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

public class Knn {
	double [][] tfidf;
	int k;
	boolean disMeas = true;
	
	public Knn(double [][] Tfidf){
		tfidf = Tfidf;
	}
	
	public void rknn(int K, int numDocs, boolean dismeas, int docIndex, int length, HashMap<Integer, String> labels, String [] arr) {
		disMeas = dismeas;
		k = K;
		double min = Double.POSITIVE_INFINITY;
		ArrayList<Double> vals = new ArrayList<Double>();
		ArrayList<Integer> indexs = new ArrayList<Integer>();
		HashMap<String, Double> counter = new HashMap<String, Double>();
		double dist = 0;
		int index = -1;
		for(int i = 0; i < numDocs; i++) {
			dist = 0;
			for(int j = 0; j < length; j++) {
				dist += distance(tfidf[i][j], tfidf [docIndex][j]);
			}
			if(dist < min) {
				vals.add(dist);
				indexs.add(i);
				
				if(vals.size() > k) {
					double tempmin = Double.NEGATIVE_INFINITY;
					for(int h = 0; h < vals.size(); h++) {
						if(vals.get(h) > tempmin) {
							tempmin = vals.get(h);
							index = h;
						}
					}
					vals.remove(index);
					indexs.remove(index);
				}
				
				for(int h = 0; h < vals.size(); h++) {
					if(vals.get(h) > min) {
						min = vals.get(h);
					}
				}
			}
		}
		
		for(int a = 0; a < indexs.size(); a++) {
			String countl = labels.get(indexs.get(a));
			if(counter.containsKey(countl)) {
				counter.replace(countl, counter.get(countl) + 1);
			}
			else {
				counter.put(countl, (double) 1);
			}
		}
		
		double max = 0;
		for(Entry<String, Double> entry : counter.entrySet()) {
			String key = entry.getKey();
			Double value = entry.getValue();
			double temp = value;
			if(temp > max) {
				arr[docIndex - 24] = key;
				max = temp;
			}
		}
		
		int den = counter.size();
		for(Entry<String, Double> entry : counter.entrySet()) {
			String key = entry.getKey();
			Double value = entry.getValue();
			double newval = value/k;
			counter.replace(key, newval);
		}
		ArrayList<String> toP = new ArrayList<String>();
		for(Entry<String, Double> entry : counter.entrySet()) {
			String key = entry.getKey();
			Double value = entry.getValue();
			value = value * 100;
			String valueStr = String.format("%.2f%%", value);
			String temp2 = valueStr + " " + key;
			toP.add(temp2);
		}
		
		String temp = toP.toString();
		labels.put(docIndex, temp);
	}
	
	public double distance(double x, double y) {
	      return disMeas ? Distance.D1(x, y) : Distance.D2(x, y);
	   }
	
	public static class Distance {
		public static double D1(double x, double y) {
			double dist = 0; 
		    dist += Math.abs(x - y);
		    return dist;
		}
		      
		public static double D2(double x, double y) {
			double dist = 0;
		    dist += Math.abs((x - y) * (x - y));
		    return dist;
		}
	}
}
