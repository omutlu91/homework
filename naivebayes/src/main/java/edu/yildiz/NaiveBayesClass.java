package edu.yildiz;

import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;

public class NaiveBayesClass {


    public static final String INSULT = "1";
    public static final String NON_INSULT = "0";

    public void naiveBayes(String trainFilename, String stopwordFilename, int trainInsultSelection, int trainNonInsultSelection) {
        System.out.println("**************");
        System.out.println("Train Started");
        System.out.println("**************");

        //Train Dosyasi okunur
        List<String[]> allTrainList = readCSVFile(trainFilename);
        //Stopword dosyasi okunur
        List<String[]> stopwords = readCSVFile(stopwordFilename);

        List<String[]> sampleTrainList = new ArrayList<>();
        List<String> insult = new ArrayList<>();
        List<String> nonInsult = new ArrayList<>();
        Set<String> vocublary = new HashSet<>();

        int insultClauseCount = 0;
        int nonInsultClauseCount = 0;

        //Train dosyasindaki secilen son index;
        int lastIndex = 0;
        for (int i = 1; i < allTrainList.size(); i++) {
            lastIndex++;
            String[] line = allTrainList.get(i);
            String checkString = line[2];
            //Noktalama isaretleri ve Illegal Karakterler temizlenir
            checkString = clearClause(checkString);

            //Stopword lerden temizlenir
            List<String> removedStopwordList = removeStopwords(checkString, stopwords);

            //Parametrede verilen sayilara gore insult ve non insult cumleler train dosyasindan secilir
            if (line[0].equalsIgnoreCase(INSULT) && insultClauseCount < trainInsultSelection) {
                insult.addAll(removedStopwordList);
                sampleTrainList.add(line);
                vocublary.addAll(removedStopwordList);
                insultClauseCount++;
            } else if (line[0].equalsIgnoreCase(NON_INSULT) && nonInsultClauseCount < trainNonInsultSelection) {
                nonInsult.addAll(removedStopwordList);
                sampleTrainList.add(line);
                vocublary.addAll(removedStopwordList);
                nonInsultClauseCount++;
            }
            if (insultClauseCount == trainInsultSelection + 1 && nonInsultClauseCount == trainNonInsultSelection + 1) {
                //Eger Parametrede verilen sayilara ulasilmissa donguden cikilir
                break;
            }
        }
        System.out.println("Selected Insult Clause Count :" + insultClauseCount);
        System.out.println("Selected Non-Insult Clause Count :" + nonInsultClauseCount);

        int truePositive = 0;
        int trueNegative = 0;
        int falsePositive = 0;
        int falseNegative = 0;

        List<String> insultSuccess = new ArrayList<>();
        List<String> insultFail = new ArrayList<>();
        List<String> nonInsultSuccess = new ArrayList<>();
        List<String> nonInsultFail = new ArrayList<>();

        //Insult ve Non Insult cumlelerinin toplami kadar dongu olusturulur.
        // Her bir kelime icin olasilik bulunur
        for (int i = 0; i < trainInsultSelection + trainNonInsultSelection; i++) {
            String[] line = sampleTrainList.get(i);
            String sentence = line[2];
            String checkString = sentence;

            //Noktalama isaretleri ve Illegal Karakterler temizlenir
            checkString = clearClause(checkString);

            //Stopwordler temizlenir
            List<String> removedStopwordList = removeStopwords(checkString, stopwords);

            //P(Insult)
            BigDecimal insultSum = new BigDecimal("1");
            insultSum = insultSum.multiply(new BigDecimal(insultClauseCount)).divide(BigDecimal.valueOf(insultClauseCount + nonInsultClauseCount), 4, RoundingMode.HALF_UP);

            //P(nonInsult)
            BigDecimal nonInsultSum = new BigDecimal("1");
            nonInsultSum = nonInsultSum.multiply(new BigDecimal(nonInsultClauseCount)).divide(BigDecimal.valueOf(insultClauseCount + nonInsultClauseCount), 4, RoundingMode.HALF_UP);

            for (String word : removedStopwordList) {
                int countInstult = calculateCount(insult, word);
                //Insult Navie Bayes Hesabi yapilir
                BigDecimal insultNaive = calculateNaiveBayes(countInstult, insult.size(), vocublary.size());

                //P(word|Insult)
                insultSum = insultSum.multiply(insultNaive).setScale(10, RoundingMode.HALF_UP);

                int countNatural = calculateCount(nonInsult, word);

                //Non Insult Navie Bayes hesabi yapilir
                BigDecimal nonInsultNaive = calculateNaiveBayes(countNatural, insult.size(), vocublary.size());


                //P(word|NonInsult)
                nonInsultSum = nonInsultSum.multiply(nonInsultNaive).setScale(10, RoundingMode.HALF_UP);

                //	System.out.println("word : "+ word +" countInsult " + countInstult +" countNatural " +countNatural  +" insultNaive "+ insultNaive + " nonInsultNaive " +nonInsultNaive);
            }


            boolean isInsult = false;
            //Insult ve Noninsult navie bayes sonuclari karsilastirilir
            if (insultSum.compareTo(nonInsultSum) > 0) {
                isInsult = true;
            }
            //Train cumlesi gercekte insult midir
            boolean isRealInsult = line[0].equalsIgnoreCase("1");

            //Dogruluk oranlari hesaplanir
            if (isRealInsult) {
                if (isInsult) {
                    if (truePositive < 11) {
                        insultSuccess.add(sentence);
                    }
                    truePositive++;
                } else {
                    if (falseNegative < 11) {
                        insultFail.add(sentence);
                    }
                    falseNegative++;
                }
            } else {
                if (isInsult) {
                    if (falsePositive < 11) {
                        nonInsultFail.add(sentence);
                    }
                    falsePositive++;
                } else {
                    if (trueNegative < 11) {
                        nonInsultSuccess.add(sentence);
                    }
                    trueNegative++;
                }
            }
            //	System.out.println("insultSum " + insultSum + " nonInsultSum " + nonInsultSum + " is Insult ? "+ isInsult + " real Insult" + isRealInsult);

        }

        System.out.println("*******");
        System.out.println("SAMPLES");
        System.out.println("*******");
        System.out.println("Prediction of Insult Success started :");
        System.out.println("*******");
        for (String success : insultSuccess) {
            System.out.println(success);
        }
        System.out.println("*******");
        System.out.println("Prediction of Insult Success ended :");
        System.out.println("*******");
        System.out.println("Prediction of Non Insult Success started :");
        System.out.println("*******");
        for (String success : nonInsultSuccess) {
            System.out.println(success);
        }
        System.out.println("*******");
        System.out.println("Prediction of Non Insult Success ended :");
        System.out.println("*******");
        System.out.println("Prediction of Insult Fail started :");
        System.out.println("*******");
        for (String fail : insultFail) {
            System.out.println(fail);
        }
        System.out.println("*******");
        System.out.println("Prediction of Insult Fail ended :");
        System.out.println("*******");
        System.out.println("Prediction of Non Insult Fail started :");
        System.out.println("*******");
        for (String fail : nonInsultFail) {
            System.out.println(fail);
        }
        System.out.println("*******");
        System.out.println("Prediction of Non Insult Fail ended :");
        System.out.println("*******");

        System.out.println("trueNegative " + trueNegative + " falsePositive " + falsePositive + " falseNegative " + falseNegative + " truePositive " + truePositive);


        BigDecimal precision = new BigDecimal(truePositive).divide(new BigDecimal(truePositive).add(new BigDecimal(falsePositive)), 2, RoundingMode.HALF_UP).setScale(2, RoundingMode.HALF_UP);
        BigDecimal recall = new BigDecimal(truePositive).divide(new BigDecimal(truePositive).add(new BigDecimal(falseNegative)), 2, RoundingMode.HALF_UP).setScale(2, RoundingMode.HALF_UP);
        BigDecimal fscore = new BigDecimal("2").multiply(precision).multiply(recall).divide(precision.add(recall), 2, RoundingMode.HALF_UP);


        System.out.println("----------------------------------------------------------------");
        System.out.println("Precision : " + precision + " Recall : " + recall + " F-Score : " + fscore);
        System.out.println("----------------------------------------------------------------");


        System.out.println("*****************************************************************");
        System.out.println("*************************TEST CASE*******************************");
        System.out.println("*****************************************************************");
        //Insult ve Non Insult toplamlarinin yuzde 10 u kadar test cumleleri rastgele secilir
        int testSelection = (trainInsultSelection + trainNonInsultSelection) * 10 / 100;
        System.out.println(testSelection + " test sentences will be select");
        List<String[]> sampleTestList = new ArrayList<>();
        for (int i = testSelection; i < allTrainList.size(); i++) {
            String[] line = allTrainList.get(i);
            sampleTestList.add(line);
        }

        int testTruePositive = 0;
        int testTrueNegative = 0;
        int testFalsePositive = 0;
        int testFalseNegative = 0;

        List<String> testInsultSuccess = new ArrayList<>();
        List<String> testInsultFail = new ArrayList<>();
        List<String> testNonInsultSuccess = new ArrayList<>();
        List<String> testNonInsultFail = new ArrayList<>();

        for (String[] testLine : sampleTestList) {
            String checkString = testLine[2];
            String testSentence = checkString;

            //Noktalama isaretleri ve Illegal Karakterler temizlenir
            checkString = clearClause(checkString);

            //Stopwordler temizlenir
            List<String> removedStopwordList = removeStopwords(checkString, stopwords);

            //P(Insult)
            BigDecimal testInsultSum = new BigDecimal("1");
            testInsultSum = testInsultSum.multiply(new BigDecimal(insultClauseCount)).divide(BigDecimal.valueOf(insultClauseCount + nonInsultClauseCount), 4, RoundingMode.HALF_UP);

            //P(nonInsult)
            BigDecimal testNonInsultSum = new BigDecimal("1");
            testNonInsultSum = testNonInsultSum.multiply(new BigDecimal(nonInsultClauseCount)).divide(BigDecimal.valueOf(insultClauseCount + nonInsultClauseCount), 4, RoundingMode.HALF_UP);

            for (String testWord : removedStopwordList) {
                int countInstult = calculateCount(insult, testWord);
                //Insult Navie Bayes Hesabi yapilir
                BigDecimal testInsultNaive = calculateNaiveBayes(countInstult, insult.size(), vocublary.size());

                //P(testWord|Insult)
                testInsultSum = testInsultSum.multiply(testInsultNaive).setScale(10, RoundingMode.HALF_UP);

                int countNatural = calculateCount(nonInsult, testWord);

                //Non Insult Navie Bayes hesabi yapilir
                BigDecimal testNonInsultNaive = calculateNaiveBayes(countNatural, insult.size(), vocublary.size());


                //P(testWord|NonInsult)
                testNonInsultSum = testNonInsultSum.multiply(testNonInsultNaive).setScale(10, RoundingMode.HALF_UP);
                //System.out.println("testWord : "+ testWord +" countInsult " + countInstult +" countNatural " +countNatural  +" testInsultNaive "+ testInsultNaive + " testNonInsultNaive " +testNonInsultNaive);
            }

            boolean testIsInsult = false;
            //Insult ve Noninsult navie bayes sonuclari karsilastirilir
            if (testInsultSum.compareTo(testNonInsultSum) > 0) {
                testIsInsult = true;
            }
            //Train cumlesi gercekte insult midir
            boolean testIsRealInsult = testLine[0].equalsIgnoreCase("1");

            //Dogruluk oranlari hesaplanir
            if (testIsRealInsult) {
                if (testIsInsult) {
                    if (testTruePositive < 11) {
                        testInsultSuccess.add(testSentence);
                    }
                    testTruePositive++;
                } else {
                    if (testFalseNegative < 11) {
                        testInsultFail.add(testSentence);
                    }
                    testFalseNegative++;
                }
            } else {
                if (testIsInsult) {
                    if (testFalsePositive < 11) {
                        testNonInsultFail.add(testSentence);
                    }
                    testFalsePositive++;
                } else {
                    if (testTrueNegative < 11) {
                        testNonInsultSuccess.add(testSentence);
                    }
                    testTrueNegative++;
                }
            }

            //  System.out.println("insultSum " + testInsultSum + " nonInsultSum " + testNonInsultSum + " is Insult ? "+ testIsInsult + " real Insult" + testIsRealInsult);
        }


        BigDecimal testPrecision = new BigDecimal(testTruePositive).divide(new BigDecimal(testTruePositive).add(new BigDecimal(testFalsePositive)), 2, RoundingMode.HALF_UP).setScale(2, RoundingMode.HALF_UP);
        BigDecimal testRecall = new BigDecimal(testTruePositive).divide(new BigDecimal(testTruePositive).add(new BigDecimal(testFalseNegative)), 2, RoundingMode.HALF_UP).setScale(2, RoundingMode.HALF_UP);
        BigDecimal testFscore = new BigDecimal("2").multiply(testPrecision).multiply(testRecall).divide(testPrecision.add(testRecall), 2, RoundingMode.HALF_UP);

        System.out.println("----------------------------------------------------------------");
        System.out.println("Test Precision : " + testPrecision + " Test Recall : " + testRecall + " Test F-Score : " + testFscore);
        System.out.println("----------------------------------------------------------------");

    }

    private static BigDecimal calculateNaiveBayes(int count, int size, int vocublarySize) {
        BigDecimal firstPhase = new BigDecimal(count + 1);
        BigDecimal secondPhase = new BigDecimal(size + vocublarySize);
        BigDecimal result = firstPhase.divide(secondPhase, 4, RoundingMode.HALF_UP);
        return result;
    }

    private static int calculateCount(List<String> list, String key) {
        int result = 0;
        for (String item : list) {
            if (item.equalsIgnoreCase(key)) {
                result++;
            }
        }

        return result;
    }

    private static List<String> removeStopwords(String checkString, List<String[]> stopwords) {
        List<String> result = new ArrayList<>();
        String[] lineArray = checkString.split(" ");

        for (String check : lineArray) {
            boolean isEqual = false;
            for (String[] stopword : stopwords) {
                if (check.equalsIgnoreCase(stopword[0]) || "".equalsIgnoreCase(check)) {
                    isEqual = true;
                }
                if (isEqual) {
                    break;
                }
            }
            if (!isEqual) {
                result.add(check.toLowerCase());
            }
        }
        return result;
    }

    private static String clearClause(String checkString) {
        checkString = checkString.replace("\"", "")
                .replace(".", "")
                .replace(":", "")
                .replace(";", "")
                .replace("'", "")
                .replace("\\xa0", "");
        checkString = checkString.replaceAll("\\p{Punct}", "");
        return checkString;
    }

    public List<String[]> readCSVFile(String fileName) {

        String line;
        String cvsSplitBy = ",";
        List<String[]> allLineList = new ArrayList<>();

        fileName = "/" + fileName;

        InputStream resourceAsStream = getClass().getResourceAsStream(fileName);

        try (BufferedReader br = new BufferedReader(new InputStreamReader(resourceAsStream))) {

            while ((line = br.readLine()) != null) {
                String[] lineArray = line.split(cvsSplitBy);
                allLineList.add(lineArray);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return allLineList;
    }

    public static void main(String[] args) throws IOException {

        try {
            Scanner in = new Scanner(System.in);
            System.out.println("Lutfen Egitilecek Insult Sayisini Giriniz");
            Integer trainInsultSelection = Integer.valueOf(in.nextLine());

            System.out.println("Lutfen Egitilecek Non Insult Sayisini Giriniz");
            Integer trainNonInsultSelection = Integer.valueOf(in.nextLine());

            NaiveBayesClass n = new NaiveBayesClass();
            n.naiveBayes("train.csv", "stopwords.csv", trainInsultSelection, trainNonInsultSelection);
        } catch (Exception e) {
            System.out.println("Hatalı Parametre");
            e.fillInStackTrace();
        }


    }
}