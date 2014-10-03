/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    UnskewedBagging.java
 *    Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.meta;

import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.*;
import weka.core.AdditionalMeasureProducer;

/**
 <!-- globalinfo-start -->
 * Class for bagging a classifier to reduce variance. Can do classification and regression depending on the base learner. <br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Leo Breiman (1996). UnskewedBagging predictors. Machine Learning. 24(2):123-140.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Breiman1996,
 *    author = {Leo Breiman},
 *    journal = {Machine Learning},
 *    number = {2},
 *    pages = {123-140},
 *    title = {UnskewedBagging predictors},
 *    volume = {24},
 *    year = {1996}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -P
 *  Size of each bag, as a percentage of the
 *  training set size. (default 100)</pre>
 * 
 * <pre> -O
 *  Calculate the out of bag error.</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.REPTree)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.trees.REPTree:
 * </pre>
 * 
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf (default 2).</pre>
 * 
 * <pre> -V &lt;minimum variance for split&gt;
 *  Set minimum numeric class variance proportion
 *  of train variance for split (default 1e-3).</pre>
 * 
 * <pre> -N &lt;number of folds&gt;
 *  Number of folds for reduced error pruning (default 3).</pre>
 * 
 * <pre> -S &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 * 
 * <pre> -P
 *  No pruning.</pre>
 * 
 * <pre> -L
 *  Maximum tree depth (default -1, no maximum)</pre>
 * 
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author Bernhard Pfahringer (bernhard@cs.waikato.ac.nz)
 * @version $Revision: 1.0 $
 */
public class UnskewedBagging
  extends RandomizableIteratedSingleClassifierEnhancer 
  implements WeightedInstancesHandler,
             TechnicalInformationHandler, AdditionalMeasureProducer {

  /** for serialization */
  static final long serialVersionUID = -505879962237199703L;
  
  /** Maps the names of additional measures to their summed values over all classier iterations */
  protected HashMap<String, Double> m_baseClassifierMeasureValues = new HashMap<String, Double>();
  
  /** The size of each bag sample, as a percentage of the training size */
  protected double m_BagSizePercent = 1.0;
  
  /** Use Roughly Balanced Bagging (RBB) */
  protected boolean m_UseRoughlyBalancedBagging = false;
  
  /** Roughly Balanced Bagging (RBB) parameter, chance of drawing a minority class example on each instance draw */
  protected double m_RoughlyBalancedBaggingMinorityChance = 0.5;
  
  /** Don't sample with replacement on the minority class */
  protected boolean m_NoReplacementMinorityClass = false;
  
  /** Don't sample with replacement on the majority class */
  protected boolean m_NoReplacementMajorityClass = false;

  /** Whether to calculate the out of bag error */
  protected boolean m_CalcOutOfBag = false;

  /** The out of bag error that has been calculated */
  protected double m_OutOfBagError;  
    
  /**
   * Constructor.
   */
  public UnskewedBagging() {
    
    m_Classifier = new weka.classifiers.trees.J48();
  }
  
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
 
    return "Class for bagging a classifier to reduce variance. Can do classification "
      + "and regression depending on the base learner. \n\n"
      + "For more information, see\n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Leo Breiman");
    result.setValue(Field.YEAR, "1996");
    result.setValue(Field.TITLE, "UnderBag predictors");
    result.setValue(Field.JOURNAL, "Machine Learning");
    result.setValue(Field.VOLUME, "24");
    result.setValue(Field.NUMBER, "2");
    result.setValue(Field.PAGES, "123-140");
    
    return result;
  }

  /**
   * String describing default classifier.
   * 
   * @return the default classifier classname
   */
  protected String defaultClassifierString() {
    
    return "weka.classifiers.trees.REPTree";
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(6);

    newVector.addElement(new Option(
              "\tSize of each bag, as a percentage of the\n" 
              + "\ttraining set size. (default 100)",
              "P", 1, "-P"));
    newVector.addElement(new Option(
              "\tUse Roughly Balanced Bagging (RBB).",
              "B", 0, "-B"));
    newVector.addElement(new Option(
              "\tRoughly Balanced Bagging (RBB) parameter,\n"
              + "\tchance of drawing a minority class example\n"
              + "\ton each instance draw. (default 0.5)",
              "C", 1, "-C"));
    newVector.addElement(new Option(
              "\tDon't sample with replacement on the minority class.",
              "r", 0, "-r"));
    newVector.addElement(new Option(
              "\tDon't sample with replacement on the majority class.",
              "R", 0, "-R"));
    newVector.addElement(new Option(
              "\tCalculate the out of bag error.",
              "O", 0, "-O"));

    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }
    return newVector.elements();
  }


  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -P
   *  Size of each bag, as a percentage of the
   *  training set size. (default 100)</pre>
   * 
   * <pre> -O
   *  Calculate the out of bag error.</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -I &lt;num&gt;
   *  Number of iterations.
   *  (default 10)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.trees.REPTree)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.trees.REPTree:
   * </pre>
   * 
   * <pre> -M &lt;minimum number of instances&gt;
   *  Set minimum number of instances per leaf (default 2).</pre>
   * 
   * <pre> -V &lt;minimum variance for split&gt;
   *  Set minimum numeric class variance proportion
   *  of train variance for split (default 1e-3).</pre>
   * 
   * <pre> -N &lt;number of folds&gt;
   *  Number of folds for reduced error pruning (default 3).</pre>
   * 
   * <pre> -S &lt;seed&gt;
   *  Seed for random data shuffling (default 1).</pre>
   * 
   * <pre> -P
   *  No pruning.</pre>
   * 
   * <pre> -L
   *  Maximum tree depth (default -1, no maximum)</pre>
   * 
   <!-- options-end -->
   *
   * Options after -- are passed to the designated classifier.<p>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    String bagSize = Utils.getOption('P', options);
    if (bagSize.length() != 0) {
      setBagSizePercent(Double.parseDouble(bagSize));
    } else {
      setBagSizePercent(1.0);
    }
    
    setUseRoughlyBalancedBagging(Utils.getFlag('B', options));
    
    String roughlyBalancedBaggingMinorityChance = Utils.getOption('C', options);
    if (roughlyBalancedBaggingMinorityChance.length() != 0)
        setRoughlyBalancedBaggingMinorityChance(Double.parseDouble(roughlyBalancedBaggingMinorityChance));
    
    setNoReplacementMinorityClass(Utils.getFlag('r', options));
    setNoReplacementMajorityClass(Utils.getFlag('R', options));

    setCalcOutOfBag(Utils.getFlag('O', options));

    super.setOptions(options);
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {


    String [] superOptions = super.getOptions();
    String [] options = new String [superOptions.length + 8];

    int current = 0;
    options[current++] = "-P";
    options[current++] = "" + getBagSizePercent();
    
    if (getUseRoughlyBalancedBagging())
        options[current++] = "-B";
    
    options[current++] = "-C";
    options[current++] = "" + getRoughlyBalancedBaggingMinorityChance();
    
    if (getNoReplacementMinorityClass())
        options[current++] = "-r";
    if (getNoReplacementMajorityClass())
        options[current++] = "-R";

    if (getCalcOutOfBag()) { 
      options[current++] = "-O";
    }

    System.arraycopy(superOptions, 0, options, current, 
		     superOptions.length);

    current += superOptions.length;
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String bagSizePercentTipText() {
    return "Size of each bag, as a percentage of the training set size.";
  }

  /**
   * Gets the size of each bag, as a percentage of the training set size.
   *
   * @return the bag size, as a percentage.
   */
  public double getBagSizePercent() {

    return m_BagSizePercent;
  }
  
  /**
   * Sets the size of each bag, as a percentage of the training set size.
   *
   * @param newBagSizePercent the bag size, as a percentage.
   */
  public void setBagSizePercent(double newBagSizePercent) {

    m_BagSizePercent = newBagSizePercent;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String useRoughlyBalancedBagging()
  {
    return "Roughly Balanced Bagging (RBB), set to true to use this technique.";
  }

  /**
   * Gets if Roughly Balanced Bagging is being used.
   *
   * @return if roughly balanced bagging is being used.
   */
  public boolean getUseRoughlyBalancedBagging()
  {
    return m_UseRoughlyBalancedBagging;
  }
  
  /**
   * Sets Roughly Balanced Bagging to be used.
   *
   * @param newUseRoughlyBalancedBagging is set to true to use Roughly Balanced Bagging.
   */
  public void setUseRoughlyBalancedBagging(boolean newUseRoughlyBalancedBagging)
  {
    m_UseRoughlyBalancedBagging = newUseRoughlyBalancedBagging;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String roughlyBalancedBaggingMinorityChance()
  {
    return "Roughly Balanced Bagging (RBB) parameter, chance of drawing a minority class example on each instance draw.";
  }

  /**
   * Gets the chance of drawing a minority class example with Roughly Balanced Bagging.
   *
   * @return the minority chance.
   */
  public double getRoughlyBalancedBaggingMinorityChance()
  {
    return m_RoughlyBalancedBaggingMinorityChance;
  }
  
  /**
   * Sets the chance of drawing a minority class example with Roughly Balanced Bagging.
   *
   * @param newRoughlyBalancedBaggingMinorityChance the new minority chance.
   */
  public void setRoughlyBalancedBaggingMinorityChance(double newRoughlyBalancedBaggingMinorityChance)
  {
    m_RoughlyBalancedBaggingMinorityChance = newRoughlyBalancedBaggingMinorityChance;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String noReplacementMinorityClass()
  {
    return "Don't sample with replacement on the minority class.";
  }

  /**
   * Gets if replacement is being used on the minority class.
   *
   * @return if replacment is being used on the minority class.
   */
  public boolean getNoReplacementMinorityClass()
  {
    return m_NoReplacementMinorityClass;
  }
  
  /**
   * Sets if replacement is used on the minority class.
   *
   * @param newNoReplacementMinorityClass is set to true to turn replacement off.
   */
  public void setNoReplacementMinorityClass(boolean newNoReplacementMinorityClass)
  {
      m_NoReplacementMinorityClass = newNoReplacementMinorityClass;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String noReplacementMajorityClass()
  {
    return "Don't sample with replacement on the majority class.";
  }

  /**
   * Gets if replacement is being used on the majority class.
   *
   * @return if replacment is being used on the majority class.
   */
  public boolean getNoReplacementMajorityClass()
  {
    return m_NoReplacementMajorityClass;
  }
  
  /**
   * Sets if replacement is used on the majority class.
   *
   * @param newNoReplacementMajorityClass is set to true to turn replacement off.
   */
  public void setNoReplacementMajorityClass(boolean newNoReplacementMajorityClass)
  {
      m_NoReplacementMajorityClass = newNoReplacementMajorityClass;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String calcOutOfBagTipText() {
    return "Whether the out-of-bag error is calculated.";
  }

  /**
   * Set whether the out of bag error is calculated.
   *
   * @param calcOutOfBag whether to calculate the out of bag error
   */
  public void setCalcOutOfBag(boolean calcOutOfBag) {

    m_CalcOutOfBag = calcOutOfBag;
  }

  /**
   * Get whether the out of bag error is calculated.
   *
   * @return whether the out of bag error is calculated
   */
  public boolean getCalcOutOfBag() {

    return m_CalcOutOfBag;
  }

  
  /**
   * UnskewedBagging method.
   *
   * @param data the training data to be used for generating the
   * bagged classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances data) throws Exception
  {
    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    super.buildClassifier(data);

    Random random = new Random(m_Seed);
    
    List<Instance> minorityClassInstances = new ArrayList<Instance>();
    List<Instance> majorityClassInstances = new ArrayList<Instance>();
    for(int i = 0; i < data.numInstances(); i++) {
      Instance instance = data.instance(i);
      if (instance.classValue() > 0.0) {
	majorityClassInstances.add(instance);
      } else {
	minorityClassInstances.add(instance);
      }
    }

    if (minorityClassInstances.size() > majorityClassInstances.size()) {
      // swap to make minorityClassInstances the minority class
      List<Instance> temp = minorityClassInstances;
      minorityClassInstances = majorityClassInstances;
      majorityClassInstances = temp;
    }

    int minorityClassGoal = minorityClassInstances.size(); // number of minority class examples to draw
    int majorityClassGoal; // number of majority class examples to draw
    
    // if true, use Roughly Balanced Bagging
    if (m_UseRoughlyBalancedBagging)
        majorityClassGoal = roughlyBalancedBaggingMajorityClassCountCalculation(minorityClassGoal, m_RoughlyBalancedBaggingMinorityChance, random.nextInt());
    // otherwise using UnderBagging
    else
        majorityClassGoal = (int) Math.round( getBagSizePercent() * minorityClassGoal);
    
    // checks to make sure that if not using replacement there are enough instances
    if (m_NoReplacementMinorityClass && minorityClassGoal > minorityClassInstances.size())
        minorityClassGoal = minorityClassInstances.size();
    if (m_NoReplacementMajorityClass && majorityClassGoal > majorityClassInstances.size())
        majorityClassGoal = majorityClassInstances.size();
    
    for (int j = 0; j < m_Classifiers.length; j++)
    {
      Instances bagData = new Instances(data,0);
      for(int i = 0; i < minorityClassGoal; i++)
      {
          int nextMinorityInstance = random.nextInt(minorityClassInstances.size());
          bagData.add(minorityClassInstances.get(nextMinorityInstance));
          if (m_NoReplacementMinorityClass)
              minorityClassInstances.remove(nextMinorityInstance);
      }
      for(int i = 0; i < majorityClassGoal; i++)
      {
          int nextMajorityInstance = random.nextInt(majorityClassInstances.size());
          bagData.add(majorityClassInstances.get(nextMajorityInstance));
          if (m_NoReplacementMinorityClass)
              minorityClassInstances.remove(nextMajorityInstance);
      }

      if (m_Classifier instanceof Randomizable) {
	((Randomizable) m_Classifiers[j]).setSeed(random.nextInt());
      }
      
      // build the classifier
      m_Classifiers[j].buildClassifier(bagData);
      
      // set the additional measure parameters, if present
      if (m_Classifiers[j] instanceof AdditionalMeasureProducer)
      {
          // cast the current classifer to allow access to the additional measures interface methods
          AdditionalMeasureProducer currentClassifier = (AdditionalMeasureProducer)m_Classifiers[j];
          // set the additional measures names variable
          Enumeration currentClassifierMeasureNames = currentClassifier.enumerateMeasures();
          while(currentClassifierMeasureNames.hasMoreElements())
          {
              String currentMeasure = (String)currentClassifierMeasureNames.nextElement();
              if (currentMeasure.startsWith("measure"))
              {
                  if (m_baseClassifierMeasureValues.containsKey(currentMeasure.toLowerCase()))
                    m_baseClassifierMeasureValues.put(currentMeasure.toLowerCase(), currentClassifier.getMeasure(currentMeasure) + m_baseClassifierMeasureValues.get(currentMeasure.toLowerCase()));
                  else
                    m_baseClassifierMeasureValues.put(currentMeasure.toLowerCase(), currentClassifier.getMeasure(currentMeasure));
              }
          }
      }
    }
    
  }

  /**
   * Calculates the class membership probabilities for the given test
   * instance.
   *
   * @param instance the instance to be classified
   * @return preedicted class probability distribution
   * @throws Exception if distribution can't be computed successfully 
   */
  public double[] distributionForInstance(Instance instance) throws Exception {

    double [] sums = new double [instance.numClasses()], newProbs; 
    
    for (int i = 0; i < m_NumIterations; i++) {
      if (instance.classAttribute().isNumeric() == true) {
	sums[0] += m_Classifiers[i].classifyInstance(instance);
      } else {
	newProbs = m_Classifiers[i].distributionForInstance(instance);
	for (int j = 0; j < newProbs.length; j++)
	  sums[j] += newProbs[j];
      }
    }
    if (instance.classAttribute().isNumeric() == true) {
      sums[0] /= (double)m_NumIterations;
      return sums;
    } else if (Utils.eq(Utils.sum(sums), 0)) {
      return sums;
    } else {
      Utils.normalize(sums);
      return sums;
    }
  }
 
    public Enumeration enumerateMeasures()
    {
        if (m_Classifier instanceof AdditionalMeasureProducer)
            return ((AdditionalMeasureProducer)m_Classifier).enumerateMeasures();
        else
            return new Vector(0).elements();
    }

    public double getMeasure(String additionalMeasureName)
    {
        if (m_Classifier instanceof AdditionalMeasureProducer)
        {
            if (m_baseClassifierMeasureValues.containsKey(additionalMeasureName.toLowerCase()))
                return m_baseClassifierMeasureValues.get(additionalMeasureName.toLowerCase());
            else
                throw new IllegalArgumentException("The additional measure, " + additionalMeasureName + ", could not be found.");
        }
        else
            throw new IllegalArgumentException("Additional measures not supported by base classifier.");
    }

  /**
   * Returns description of the bagged classifier.
   *
   * @return description of the bagged classifier as a string
   */
  public String toString() {
    
    if (m_Classifiers == null) {
      return "UnskewedBagging: No model built yet.";
    }
    StringBuffer text = new StringBuffer();
    text.append("All the base classifiers: \n\n");
    for (int i = 0; i < m_Classifiers.length; i++)
      text.append(m_Classifiers[i].toString() + "\n\n");
    
    if (m_CalcOutOfBag) {
      text.append("Out of bag error: "
		  + Utils.doubleToString(m_OutOfBagError, 4)
		  + "\n\n");
    }

    return text.toString();
  }

  public String getRevision() {
    //return RevisionUtils.extract("$Revision: 1.41 $");
    return "1.0";
  }
  
  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {
    runClassifier(new UnskewedBagging(), argv);
  }
  
  private int roughlyBalancedBaggingMajorityClassCountCalculation(int minorityClassGoal, double minorityClassChance, int randomSeed)
  {
      // check the minority chance is within the 0->1 range, if it is not then return the minority goal
      if (minorityClassChance <= 0.0 || minorityClassChance > 1.0)
          return minorityClassGoal;
      
      int minorityCount = 0;
      int majorityCount = 0;
      
      Random random = new Random(randomSeed);
      
      while (minorityCount < minorityClassGoal)
      {
          if (random.nextDouble() < minorityClassChance)
              minorityCount++;
          else
              majorityCount++;
      }
      
      return majorityCount;
  }
}
