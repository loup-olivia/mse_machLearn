# PW_09: Classification Trees and Random Forests
## Students
- Liechti Matthieu
- Loup Olivia
## Date
17.11.24

## Ex 1 : Classification trees
### Q1.1: At which frequencies does the electrical activity mostly occur? 

### Q1.2: Can you easily distinguish between classes visually? What can you say about the inter- vs intra-class variability?   
### Q1.3: Describe both of these criteria. What does a gini impurity of 0 means? What does an entropy of 1 mean?
### Q1.4: This problem suffers from a common issue in machine learning. What is this problem called? What could be its causes? How can it be resolved?
### Q1.5: Use the visualization of this tree to show and explain:
- What is a node? What is an edge? What is a leaf?
- What are the two additional hyperparameters doing? Do you think that both 
  are necessary in this particular case (min_samples_leaf = 20, max_depth= 4)? Why?
- What does the color of each node represent?
### Q1.6: Choose one of the nodes. Explain precisely the information given on each line of text in this node.
### Q1.7: Does model 2 still have the same problem as model 1? Explain based on the classification reports and the confusion matrices.
### Q1.8: One of the class seems more difficult to predict than others? Which one? Where could this difficulty come from in your opinion?
### Q1.9: What does this hyperparameter do? Explain giving examples from this dataset.
### Q1.10: Compare result

## Ex 2 : Random forests 
### Q2.1: For each of the hyperparameter: Is there a range of value giving particularly good results? Or particularly bad results?
- **'max_depth'**: 
    - Good result : [0;10]
    - Bad result : [30;50]
- **'max_features'**: (some similarities  with all value)
    - Good result : > 0 
    - Bad result :[0]
- **'min_samples_leaf'**: (some similarities  with all value)
    - Good result : >15 
    - Bad result : [>5;<15]
- **'n_estimators'**: (Have some bad results at some few points)
    - Good result : > 15
    - Bad result : < 15
### Q2.2: These representations give valuable information about hyperparameters. It is nevertheless insufficient. What are/is the main problem(s) with those graphs in your opinion?
In two case, these reprensentations don't give enough differences between results. With this, it's difficult to see the real bests parameters.
### Q2.3: What do the following plots represent?
It's the mean test score (mean of number of good value predict) depend of the value and hyperparameter type. 
### Q2.4: What do the white spots (=empty spots) in the heatmaps mean?
The result fewer than the representation on the array.
### Q2.5: How do those plots address the limitations of the previous visualizations?
For the most visualisable values. 
### Q2.6: What is grid search? Explain by giving real examples from this specific task.
Grid search is used to try out different parameter/hyperparameter to choose the best one or the best combinations of parameters (if more than one).\
In example with this case, we have differents hyperparameter : 'max_depth', 'max_features','n_estimators', 'min_samples_leaf'. We should test all combinations, to find best fit of values to create model.

### Q2.7: Use the plots above to narrow the range of hyperparameters you want to explore. Which values did you choose to test for each parameter? Justify your choices.
the plots above "*Pair-wise Comparison of Hyperparameters*"show the result of mean test score depend of hyperparameter. This show us than they have more good result :
- 'max_depth': better around [3;7]
- 'min_samples_leaf': better around [21;24] (less obvious than 'max_depth')
- 'max_features' : depend most of other hyperparameters, with 'n_estimators'and 'min_samples_leaf', this is better around [15;20] (no difference with 'max_depth')
- 'n_estimators' : 181 (no difference with 'n_estimators', good result with this value and 'max_features' 
or 'min_samples_leaf')
- 'min_samples_leaf': [1,18,19] (not big stability of good value )

### Q2.8: Which value did you choose for each hyperparameter?
The values of hyperparameters are :
- **'max_depth'**: 7 
- **'max_features'**: 23
- **'min_samples_leaf'**: 18
- **'n_estimators'**: 181

This values depend of the graphics of the pair-wire comparison of hyperparmeters and the results of the previous part after several test.
### Q2.10: The test set should be used only at this stage, and it is theoretically important not to change the hyperparameters based on the performance on the test set. Why?
Because this parameters depend also of training set, change parameters may not correpsond with previous dataset. It must be conciderate like biais to change on the test set. 

### Q2.11: Comment your results. -> How well does the model generalize on unseen data? Is a random forest better than a single classification tree in this case? What is the main challenge of this dataset? …
The features could have an impact on result because of resut of mesurable proprety.\
The resutl give :
- **Accuracy**: 93.213%
- **F1 score**: 93.461%

|            |  precision|    recall|  f1-score|   support|
|-|-|-|-|-|
|        Wake|       0.94|      0.94|      0.94|      1376|
|        NREM|       0.65|      0.90|      0.76|       271|
|         REM|       0.97|      0.93|      0.95|      2405|
||||||
|    accuracy|           |          |      0.93|      4052|
|   macro avg|       0.86|      0.92|      0.88|      4052|
|weighted avg|       0.94|      0.93|      0.93|      4052|


With the result on F1-score and accuracy, the model give a good result.
The main challenge is the number of hyperparameters and the number of good values. It could take time to find the good ones because there were severall values of hyperparameters seems corrects before.\
The most good result are with the REM classes and the worst with the NREM classes.\
In this case, the random forest is better because the score of the test set is better than classification tree.
  
### Q2.11: How is this importance calculated?
they are computed as the mean and standard deviation of accumulation of the impurity decrease within each tree.
### Q2.12: What can you conclude from this graph?
In this graph, the highter the result are, the best is for the model. In this case, the best result seems to be with the 17 firsts features in depend of the features between 8 and 12 than are lower.

## Ex 3 : Gradient boosting
### Q3.1: Two additional hyperparameters were added compared to the RandomForestClassifier. What are these hyperparameters, and what roles do they play?
### Q3.2: Comment the results. Compare these results with the ones obtained with the RandomForestClassifier. Compare more specifically the precision, the recall and the f1-score of the 'r' class obtained with GradientBoostingClassifier and RandomForestClassifier. What are your conclusions?