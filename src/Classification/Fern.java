package Classification;

import org.opencv.core.MatOfKeyPoint;

public class Fern {
	String name;
	int [] histogram;
	int size;
	int hist_size = 0;
	int max = 0;
	
	
	public Fern(int size){
		this.size = size;
		histogram = new int [size];
		for(int i=0;i<size;i++)
		{
			histogram[i]=0;
		}
	}
	public void getPercentage()
	{
	System.out.println("pravdepodobnost:" + (float)majority()/(float)max + " bodu " + majority());	
	System.out.println("pocet dvojic: " + max);
	}
	
	public void setName(String name)
	{
		this.name = name;
	}
	public int getHist_size() {
		return hist_size;
	}

	public void HistogramAdd(int index)
	{
		if(histogram[index]==0) hist_size++;
		histogram[index]++;
		max++;
	}
	
	public int majority()
	{
		int max = 0;
		int max_position = 0;
		for(int i =0;i<size;i++)
		{
			if(histogram[i]>max)
				{
				max = histogram[i];
				max_position = i;
				}
		}
		return max_position;
	}
	public void showHistogram()
	{
		
		for(int i=0;i<size;i++)
		{
			System.out.println(i + " : " + histogram[i]);
		}
	}
	/*
	 * Natrenovanie fernu - poslanie keypointov, ktore sa prejdu a podla cisel ktore maju pridaju 
	 * do histogramu inkrement
	 */
	public void train(MatOfKeyPoint k)
	{
		for(int i =0;i<k.size().height;i++)
		{
		HistogramAdd((int)k.toList().get(i).response);
		}
	}
	
	public void toBinary(String bin_num,int number)
	{	
		if(number != 0)
		{
			bin_num = bin_num  + number%2;
			number=number/2;
			toBinary(bin_num,number);
		}
		else System.out.println(bin_num);
	}
}
