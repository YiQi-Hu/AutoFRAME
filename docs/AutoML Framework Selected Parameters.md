# AutoML Framework Selected Parameters

 * [Content](#automl-framework-selected-parameters)
      * [Scikit-Learn Model](#scikit-learn-model)
         * [DecisionTreeClassifier](#decisiontreeclassifier)
         * [ExtraTreeClassifier](#extratreeclassifier)
         * [SVC](#svc)
         * [NuSVC](#nusvc)
         * [LinearSVC](#linearsvc)
         * [KNeighbors](#kneighbors)
         * [RadiusNeighbors](#radiusneighbors)
         * [LogisticRegression](#logisticregression)
         * [SGDClassifier](#sgdclassifier)
         * [RidgeClassifier](#ridgeclassifier)
         * [PassiveAggressiveClassifier](#passiveaggressiveclassifier)
         * [Perceptron](#perceptron)
         * [GaussianProcessClassifier](#gaussianprocessclassifier)
         * [AdaBoostClassifier](#adaboostclassifier)
         * [BaggingClassifier](#baggingclassifier)
         * [ExtraTreesClassifier](#extratreesclassifier)
         * [RandomForest](#randomforest)
         * [LinearDiscriminantAnalysis](#lineardiscriminantanalysis)
         * [QuadraticDiscriminantAnalysis](#quadraticdiscriminantanalysis)
         * [GaussianNB](#gaussiannb)
         * [BernoulliNB](#bernoullinb)
         * [MultinomialNB](#multinomialnb)
      * [LightGBM](#lightgbm)
         * [LGBMClassifier](#lgbmclassifier)
      * [XGBoost](#xgboost)
         * [XGBoostClassifier](#xgboostclassifier)

There are 3 types of parameters

- 0 --> float
- 1 --> integer
- 2 --> categorical

## Scikit-Learn Model

### DecisionTreeClassifier

#### Parameters

| Name                  | Type | Range                  |
| --------------------- | ---- | ---------------------- |
| criterion             | 2    | ['gini', 'entropy']    |
| max_depth             | 1    | [0, 40]                |
| min_samples_split     | 1    | [1, 100]               |
| min_samples_leaf      | 1    | [1, 100]               |
| max_features          | 2    | ['sqrt', 'log2', None] |
| max_leaf_nodes        | 1    | [-1, 100]              |
| min_impurity_decrease | 0    | [0.0, 100.0]           |

#### Example

```python
# ['gini', 10, 10, 10, 'auto', -1, 0.0] is like
[0, 10, 10, 10, 0, -1, 0.0]
```

### ExtraTreeClassifier

#### Parameters

| Name                  | Type | Range                          |
| --------------------- | ---- | ------------------------------ |
| criterion             | 2    | ['gini', 'entropy']            |
| max_depth             | 1    | [0, 40]                        |
| min_samples_split     | 1    | [1, 100]                       |
| min_samples_leaf      | 1    | [1, 100]                       |
| max_features          | 2    | ['auto', 'sqrt', 'log2', None] |
| max_leaf_nodes        | 1    | [-1, 100]                      |
| min_impurity_decrease | 0    | [0.0, 100.0]                   |

#### Example

```python
# ['gini', 10, 10, 10, 'auto', -1, 0.0] is like
[0, 10, 10, 10, 0, -1, 0.0]
```

### SVC

#### Parameters 

| Name      | Type | Range                                               |
| --------- | ---- | --------------------------------------------------- |
| C         | 0    | [0.01, 1e6]                                         |
| kernel    | 2    | ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] |
| degree    | 1    | [1, 30]                                             |
| gamma     | 0    | [1e-5, 10]                                          |
| coef0     | 0    | [0.0, 100.0]                                        |
| shrinking | 2    | [True, False]                                       |
| tol       | 0    | [1e-5, 1]                                           |

#### Example

```python
# [0.03, 'rbf', 1, 1, 0, True, 1e-3]
[0.03, 2, 1, 1, 0, 0, 1e-3]
```

### NuSVC

#### Parameters 

| Name      | Type | Range                                               |
| --------- | ---- | --------------------------------------------------- |
| nu        | 0    | [5e-3, 1]                                           |
| kernel    | 2    | ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] |
| degree    | 1    | [1, 30]                                             |
| gamma     | 0    | [1e-5, 10]                                          |
| coef0     | 0    | [0.0, 100.0]                                        |
| shrinking | 2    | [True, False]                                       |
| tol       | 0    | [1e-5, 1]                                           |

#### Example

```python
# [0.3, 'rbf', 1, 1, 0, True, 1e-3]
[0.3, 2, 1, 1, 0, 0, 1e-3]
```

### LinearSVC

#### Paramters

| Name    | Type | Range                      |
| ------- | ---- | -------------------------- |
| penalty | 2    | ['l1', 'l2']               |
| loss    | 2    | ['hinge', 'squared_hinge'] |
| dual    | 2    | [True, False]              |
| tol     | 0    | [1e-6, 1e-1]               |
| C       | 0    | [0.01, 1e6]                |

#### Examples

```python
# ['l1', 'hinge', True, 1e-5, 1.0]
[0, 0, 0, 1e-5, 1.0]
```

### KNeighbors

#### Parameters

| Name        | Type | Range                              |
| ----------- | ---- | ---------------------------------- |
| n_neighbors | 1    | [1, 50]                            |
| weights     | 2    | ['uniform', 'distance']            |
| algorithm   | 2    | ['ball_tree',  'kd_tree', 'brute'] |
| leaf_size   | 1    | [3, 100]                           |
| p           | 1    | [1, 10]                            |

### Examples

```python
# [5, 'uniform', 'ball_tree', 5, 2]
[5, 0, 0, 5, 2]
```

### RadiusNeighbors

#### Parameters

| Name   | Type | Range       |
| ------ | ---- | ----------- |
| radius | 0    | [1e-2, 1e2] |
| weights      | 2    | ['uniform', 'distance']            |
| algorithm   | 2    | ['ball_tree',  'kd_tree', 'brute'] |
| leaf_size   | 1    | [3, 100]                           |
| p           | 1    | [1, 10]                            |

### Examples

```python
# [5, 'uniform', 'ball_tree', 5, 2]
[5, 0, 0, 5, 2]
```

### LogisticRegression

#### Paramters

| Name        | Type | Range                                              |
| ----------- | ---- | -------------------------------------------------- |
| penalty     | 2    | ['l1', 'l2']                                       |
| dual        | 2    | [True, False]                                      |
| tol         | 0    | [1e-6, 1e-1]                                       |
| C           | 0    | [1e-2, 1e2]                                        |
| solver      | 2    | ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] |
| max_iter    | 1    | [1e2, 1e3]                                         |
| multi_class | 2    | ['ovr', 'multinomial']                             |

#### Example

```python
# ['l1', True, 1e-3, 1e-1, 'lbfgs', 100, 'ovr']
[0, 0, 1e-3, 1e-1, 1, 100, 0]
```

### SGDClassifier

#### Parameters

| Name          | Type | Range                                                        |
| ------------- | ---- | ------------------------------------------------------------ |
| loss          | 2    | ['hinge', 'log', 'modified_huber','squared_hinge', 'perceptron'] |
| penalty       | 2    | ['none', 'l2', 'l1', 'elasticnet']                           |
| alpha         | 0    | [1e-5, 1e-3]                                                 |
| l1_ratio      | 0    | [0.0, 1.0]                                                   |
| max_iter      | 1    | [1000, 10000]                                                |
| tol           | 0    | [1e-4, 1e-2]                                                 |
| learning_rate | 2    | ['constant', 'optimal', 'invscaling', 'adaptive']            |
| eta0          | 0    | [0.0, 10.0]                                                  |
| power_t       | 0    | [0.05, 1]                                                    |

#### Example

```python
# ['log', 'l2', 1e-4, 0.15, 1000, 1e-3, 'optimal', 0.0, 0.5]
[1, 1, 1e-4, 0.15, 1000, 1e-3, 1, 0.0, 0.5]
```

### RidgeClassifier

#### Parameters

| Name     | Type | Range                                                   |
| -------- | ---- | ------------------------------------------------------- |
| alpha    | 0    | [1e-2, 1e2]                                             |
| max_iter | 1    | [1000, 10000]                                           |
| tol      | 0    | [1e-4, 1e-2]                                            |
| solver   | 2    | [‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’] |

#### Example

```python
# [1.0, 1000, 1e-3, 'sag']
[1.0, 1000, 1e-3, 4]
```

### PassiveAggressiveClassifier

#### Parameters

| Name     | Type | Range                      |
| -------- | ---- | -------------------------- |
| C        | 0    | [0.1, 10]                  |
| max_iter | 1    | [1000, 10000]              |
| tol      | 0    | [1e-4, 1e-2]               |
| loss     | 2    | ['hinge', 'squared_hinge'] |

#### Example

```python
# [1.0, 1000, 1e-3, 'hinge']
[1.0, 1000, 1e-3, 0]
```

### Perceptron

#### Parameters

| Name     | Type | Range                            |
| -------- | ---- | -------------------------------- |
| penalty  | 2    | [None, 'l2', 'l1', 'elasticnet'] |
| alpha    | 0    | [1e-5, 1e-3]                     |
| max_iter | 1    | [1000, 10000]                    |
| tol      | 0    | [1e-4, 1e-2]                     |
| eta0     | 0    | [0.1, 10]                        |

#### Example

```python
# [None, 1e-4, 1000, 1e-3, 1.0]
[0, 1e-4, 1000, 1e-3, 1.0]
```

### GaussianProcessClassifier

#### Parameters

| Name             | Type | Range      |
| ---------------- | ---- | ---------- |
| max_iter_predict | 1    | [10, 1000] |

#### Example

```python
# [100]
[100]
```

### AdaBoostClassifier

#### Parameters 

| Name          | Type | Range                |
| ------------- | ---- | -------------------- |
| n_estimators  | 1    | [30, 500]            |
| learning_rate | 0    | [0.1, 10.]           |
| algorithm     | 2    | [‘SAMME’, ‘SAMME.R’] |

#### Example

```python
# [50, 1.0, 'SAMME']
[50, 1.0, 0]
```

### BaggingClassifier

#### Parameters

| Name         | Type | Range      |
| ------------ | ---- | ---------- |
| n_estimators | 1    | [5, 100]   |
| max_samples  | 0    | [0.0, 1.0] |
| max_features | 0    | [0.0, 1.0] |

#### Example

```python
[10, 0.5, 0.5]
```

### ExtraTreesClassifier

#### Parameters

| Name              | Type | Range               |
| ----------------- | ---- | ------------------- |
| n_estimators      | 1    | [5, 1000]           |
| critetion         | 2    | ['gini', 'entropy'] |
| max_depth             | 1    | [0, 40]                        |
| min_samples_split     | 1    | [1, 100]                       |
| min_samples_leaf      | 1    | [1, 100]                       |
| max_features          | 2    | ['sqrt', 'log2', None] |
| max_leaf_nodes        | 1    | [-1, 100]                      |
| min_impurity_decrease | 0    | [0.0, 100.0]                   |

#### Example

```python
# [10, 'gini', 10, 2, 1, 0, -1, 0.]
[10, 0, 10, 2, 1, 0, -1, 0.]
```

### RandomForest

#### Parameters

| Name         | Type | Range     |
| ------------ | ---- | --------- |
| n_estimators | 1    | [10, 1000] |
| max_depth             | 1    | [0, 40]                        |
| min_samples_split     | 1    | [1, 100]                       |
| min_samples_leaf      | 1    | [1, 100]                       |
| max_features          | 2    | ['sqrt', 'log2', None] |
| max_leaf_nodes        | 1    | [-1, 100]                      |
| min_impurity_decrease | 0    | [0.0, 100.0]                   |

#### Example

```python
# [10, 'gini', 10, 2, 1, 0, -1, 0.]
[10, 0, 10, 2, 1, 0, -1, 0.]
```

### LinearDiscriminantAnalysis

#### Parameters

| Name      | Type | Range                    |
| --------- | ---- | ------------------------ |
| solver    | 2    | ['svd', 'lsqr', 'eigen'] |
| shrinkage | 0    | [0, 1]                   |
| tol       | 0    | [1e-5, 1e-3]             |

#### Example

```python
#['svd', 0.5, 1e-4]
[0, 0.5, 1e-5]
```

### QuadraticDiscriminantAnalysis

#### Prameters

| Name      | Type | Range        |
| --------- | ---- | ------------ |
| reg_param | 0    | [0.0, 1.0]   |
| tol       | 0    | [1e-5, 1e-3] |

#### Example

```python
[0.0, 1e-4]
```

### GaussianNB

#### Parameters

| Name          | Type | Range         |
| ------------- | ---- | ------------- |
| var_smoothing | 0    | [1e-10, 1e-8] |

#### Example

```python
[1e-9]
```

### BernoulliNB

#### Parameters

| Name     | Type | Range       |
| -------- | ---- | ----------- |
| alpha    | 0    | [0.0, 10.0] |
| binarize | 0    | [0.0, 1.0]  |

#### Example

```python
[1.0, 0.0]
```

### MultinomialNB

#### Parameters

| Name  | Type | Range       |
| ----- | ---- | ----------- |
| alpha | 0    | [0.0, 10.0] |

#### Example

```python
[1.0]
```

## LightGBM

### LGBMClassifier

#### Parameters

| Name                   | Type | Range                          |
| ---------------------- | ---- | ------------------------------ |
| boosting_type          | 2    | ['gbdt', 'dart', 'goss', 'rf'] |
| num_leaves             | 1    | [10, 300]                      |
| max_depth              | 1    | [-1, 100]                      |
| learning_rate          | 0    | [0.01, 1]                      |
| n_estimators           | 1    | [10, 1000]                     |
| subsample_for_bin      | 1    | [20000, 2000000]               |
| min_split_gain         | 0    | [0., 10.]                      |
| min_child_weight       | 0    | [1e-4, 1e-2]                   |
| min_child_samples      | 1    | [1, 20]                        |
| subsample              | 0    | [0.1, 1.0]                     |
| subsample_freq         | 1    | [-1, 10]                       |
| colsample_bytree       | 0    | [0.1,  1.0]                    |
| reg_alpha              | 0    | [0.0, 1e4]                     |
| reg_lambda             | 0    | [0.0, 1e4]                     |
| max_bin                | 1    | [20, 2000]                     |
| drop_rate  (in 'dart') | 0    | [0.0, 1.0]                     |
| max_drop (in 'dart')   | 1    | [0, 500]                       |
| skip_drop (in 'dart')  | 0    | [0.0, 1.0]                     |
| top_rate (in 'goss')   | 0    | [0.0, 1.0]                     |
| other_rate(int 'goss') | 0    | [0.0, 1.0]                     |
| min_data_in_bin        | 1    | [3, 30]                        |
| sparse_threshold       | 0    | [0.1, 1.0]                     |

The meanings of the parameters please refer to [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api)

## XGBoost

### XGBoostClassifier

#### Parameters

| Name                  | Type | Range                                                |
| --------------------- | ---- | ---------------------------------------------------- |
| max_depth             | 1    | [0, 60]                                              |
| learning_rate         | 0    | [0, 1]                                               |
| n_estimators          | 1    | [50, 1000]                                           |
| booster               | 2    | [gbtree, gblinear, dart]                             |
| gamma                 | 0    | [0,  10000]                                          |
| min_child_weight      | 1    | [0, 100]                                             |
| max_delta_step        | 1    | [0, 10]                                              |
| subsample             | 0    | [0.1, 1]                                             |
| colsample_bytree      | 0    | [0.1, 1]                                             |
| colsample_bylevel     | 0    | [0.1, 1]                                             |
| reg_alpha             | 0    | [0.0, 1e4]                                           |
| reg_lambda            | 0    | [0.0, 1e4]                                           |
| tree_method           | 2    | [auto, exact, approx, hist, gpu_exact, gpu_hist]     |
| sketch_eps            | 0    | [0.003, 1]                                           |
| grow_policy           | 2    | ['depthwise', 'lossguide']                           |
| max_leaves            | 1    | [0, 100]                                             |
| max_bin               | 1    | [20, 2000]                                           |
| sample_type (dart)    | 2    | ['uniform', 'weighted']                              |
| normalize_type (dart) | 2    | ['tree', 'forest']                                   |
| rate_drop (dart)      | 0    | [0, 1]                                               |
| skip_drop (dart)      | 0    | [0, 1]                                               |
| updater (gblinear)    | 2    | ['shotgun', 'coord_descent']                         |
| feature_selector      | 2    | ['cyclic', 'shuffle', 'random', 'greedy', 'thrifty'] |

The meanings of the parameters please refer to [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) 