
<h1 align="center">
  <a>Assignment 3 – Classification Tasks (Generative Classifiers)</a>
</h1>
<h3 align="center">
  <a>Machine Learning @ Moscow State University 2021</a>
</h3>

## Idea 📓

- Consider binary and multi-class classification tasks.

- Learn to work with generative classifiers via constructing, training, and testing LDA, QDA, and Gaussian NB classifiers.

- Consider the case of d-dimensional features with d>2.

***

**The goal of all these tasks is to learn how to construct and train the particular examples of generative classifiers, namely the LDA, QDA, and GNB classifiers, which are simple but effective tools for solving (some) classification tasks.** You will have to carry out both binary and 3-class classification tasks for d-dimensional feautures with d<=4. In this exercise you consider the LDA and GNB classifiers from the linear hypothesis class (they render linear decision bounderies) as well as the QDA classifiers for which the decision boundary is not anymore an affine set in inputs (renders quadratic decision boundary).

***

## Data 📦

- [Wine Dataset]: This dataset contains informationon the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of different constituents found in each of the three types of wines. It contains 13 features and 178 examples from 3 classes. See description of variables at: 

http://archive.ics.uci.edu/ml/datasets/Wine

Taken from: Forina, M. et al, PARVUS. **Institute of Pharmaceutical and Food Analysis and Technologies**, Via Brigata Salerno, 16147 Genoa, Italy. 

The dataset is already provided in `wine.csv`:
		
	- Column **1**: class labels.
		
	- Columns **2-14**: features.

## Tasks 📝

#### Preliminaries

Please mind that it can be a good idea for those who work in Matlab to use Livescripts the way we do it in our tutorials.

- Before starting with the main code, you will need to complete the following functions:
	
    - `fit_lda`
	
    - `classify_lda`
    
    - `fit_qda`
    
    - `classify_qda`
    
    - `fit_naive_bayes_gauss`
    
    - `classify_naive_bayes_gauss`
    
    - `compute_loss`

Each function file contains the necessary instructions and starter code.

	
#### Classification

The file `main.py` will be divided as follows:
	
   **Load Data**.

	- Load the `wine.csv` dataset and select two types (classes) of wine.

1. **Part I: Binary Classification with One Feature (d=1)**:

	- *Task 0* : To better visualize the nature of the dataset, create a plot of the feature.
	
	- *Task 1* : Construct the LDA classifier.
	
	- *Task 2* : Compute the LDA training/LOOCV errors.
	
	- *Task 3* : Plot the resulting classification.
	
	- *Task 4* : Construct QDA classifier.
	
	- *Task 5* : Compute the QDA training/LOOCV errors.
	
	- *Task 6* : Plot the resulting classification.
	
	- *Task 7* : Construct  Gaussian Naive Bayes classifier.
	
	- *Task 8* : Compute the Naive Bayes Gauss training/LOOCV errors.
	
	- *Task 9* : Plot the resulting classification.
	
	- DRAW CONCLUSIONS ABOUT THE PERFORMANCE OF THE CLASSIFIERS


2. **Part II: Binary Classification with Two Features**: 
	
	- *Task 10* : Generalize LDA, QDA, GNB to the case of two features. Report their training/LOOCV errors for the stadard threshold C=1/2, plot. Compare all the classifiers via plotting their ROC curves and computing the area under ROC (AUC).
	
	
3. **Part III: 3-Classes Classification with Many Features**: 
	
	
	- *Task 11* : QDA classifier with 3 classes, training/LOOCV errors.
	

	
  
## Notes ⚠️

**Write your assignment code following the instructions given in  `main.py`**.

Please *avoid creating unnecessary scripts/function files*, as this makes the code harder to grasp in its entirety.

**Good programming rules apply**:
- Use meaningful variable names. 
- Use indentation.
- Keep your code tidy. 
- Add a minimum of comments (if you deem then necessary). 

<br>

***Work well!***
