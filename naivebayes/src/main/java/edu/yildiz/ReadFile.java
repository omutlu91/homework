package edu.yildiz;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class ReadFile {

	public String readExcel() {
		ClassLoader classLoader = getClass().getClassLoader();
		File file = new File(classLoader.getResource("a.txt").getFile());
		
		
		 StringBuilder sb = new StringBuilder();

	        try (BufferedReader br = Files.newBufferedReader(Paths.get(file.getAbsolutePath()))) {

	            // read line by line
	            String line;
	            while ((line = br.readLine()) != null) {
	                sb.append(line).append("\n");
	            }

	        } catch (IOException e) {
	            System.err.format("IOException: %s%n", e);
	        }

			return sb.toString();
		
	}

	public static void main(String[] args) {
	try {
		System.out.println(new ReadFile().readExcel());
	} catch (Exception e) {
		e.printStackTrace();
	}
		

	}

}
