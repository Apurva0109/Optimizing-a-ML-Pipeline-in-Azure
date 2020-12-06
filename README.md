# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

The dataset contains data about the direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. The goal is to use classification to predict if the client will subscribe(0/1) to a term deposit(target variable y).

__Input Variables:__ age, job(type of job), marital(marital status), education etc.

**Predict Variable(desired target):** y - Has the client subscribe to a term deposit? (binary: “1”, means “Yes”, “0” means “No”)

The best performing model was found using Scikit-Learn Logistic Regression with an accuracy of 0.9162.

## Approaches

We solved this problem with two different methods -

1. Optimize the hyperparameter of custom-coded standard Scikit-learn Logistic Regression model using HyperDrive.
1. Find the Optimal Model for the same dataset using Automated Machine Learning.

## Scikit-learn Pipeline
We need to create and load the dataset which is done by the *train.py* script.

Steps performed in train.py are:

* Load data from the given URL and create a TabularDataset to represent tabular data in CSV file.
* Clean data using clean_data function, which involves -
    * Converting categorical variable to binary using one-hot encoding.
    * Dropping missing values etc.
* Split the dataset using the “train_test_split” function in Scikit-learn into train and test in the ratio 80% and 20% respectively. 
* Use the Scikit-learn model to fit the training data.
* Compute accuracy for train data using Scikit learn model.
* Save the model in the folder outputs/model.joblib.

In this project, we used Logistic regression, a classification algorithm. It is used when the value of the target variable is categorical in nature. Logistic regression is most commonly used when the data in question has binary output, so when it belongs to one class or another, or is either a 0 or 1.

Two types of hyperparameters used are-
1. C: inverse of regularization strength
1. max_iter: maximum iteration number

We need to tune these hyperparameters, manual hyperparameter tuning is extremely time-consuming so we are using HyperDrive to find optimal hyperparameter for the logistic regression model.

Tuning hyperparameter is done in the jupyter notebook *udacity-project.ipynb*.

Steps performed in the udacity-project.ipynb for hyperparameter tuning are:
* Define search space: Dictionary is created with the set of hyperparameter values i.e C and max_iter.

    * With a smaller C value, you can specify stronger regularization.
    * max_iter specifies the maximum number of iterations taken for the solvers to converge.

```
    param_space = {
            '--C': uniform(0.05,1),
            '--max_iter': choice(20,40,60,80,100)
        }
```
* Configuring sampling: Specific values used in a hyperparameter tuning run depend on the sampling used. We have used *RandomParameterSampling*.

    Advantages of Random Parameter Sampling:

    1. Random sampling is used to randomly select value so, it eliminates sampling bias.
    2. It supports discrete and continuous hyperparameters.
    3. It reduces computation time.
```
    ps = RandomParameterSampling(param_space)
```
 

* Early termination policy: To help prevent wasting time, you can set an early termination policy that abandons runs that are unlikely to produce a better result than previously completed runs.<br>
Bandit policy stops a run if the target performance metric underperforms the best run so far by a specified margin. It is based on slack criteria and a frequency and delay interval for evaluation.
Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

* Use Estimator:
We used the SKLearn estimator to run the train.py script, which automatically includes Scikit-Learn and its dependencies in the run environment.
```
    estimator = SKLearn(source_directory='.',
                        entry_script='train.py',
                        compute_target=compute_aml_cluster
                        )
```
* Configure hyperdrive experiment:
To prepare the hyperdrive experiment, you must use a HyperDriveConfig object to configure the experiment run.<br>
HyperDriveConfig is created using the estimator, hyperparameter sampler, and early termination policy.

```

    hyperdrive_config =HyperDriveConfig(estimator=estimator,
                                hyperparameter_sampling=ps,
                                policy=policy,
                                primary_metric_name='Accuracy',
                                primary_metric_goal=   
                                PrimaryMetricGoal.MAXIMIZE,
                                max_total_runs=12,
                                max_concurrent_runs=4)
```
   *primary primary_metric_name*: The name of the primary metric reported by the experiment runs.

   *primary_metric_goal*: Either PrimaryMetricGoal.MINIMIZE or PrimaryMetricGoal.MAXIMIZE. This parameter determines if the primary metric is to be minimized or maximized when evaluating runs.

   *max_total_runs* and *max_concurrent_runs*:The maximum total number of runs to create.The maximum number of runs to execute concurrently.

* Submit experiment:
Submit your hyperdrive run to the experiment and show run details with the widget.
```
    hyperdrive_run = exp.submit(config=hyperdrive_config)
```
<img src="Hyperdrive Capture/Capture1.PNG">
<img src="Hyperdrive Capture/Capture2.PNG">

* Get the best run and metric
```
    best_run = hyperdrive_run.get_best_run_by_primary_metric()
    best_run_metrics=best_run.get_metrics()
```


## AutoML

*Automated Machine Learning (AutoML)* enables you to try multiple algorithms and preprocessing transformations with your data. This, combined with scalable cloud-based computing makes it possible to find the best performing model for your data without the huge amount of time-consuming manual trial and error that would otherwise be required.

AutoML is done in the jupyter notebook *udacity-project.ipynb*.

Steps performed in the udacity-project.ipynb for AutoML are:

* Create the dataset from the provided URL using *TabularDatasetFactory* in the notebook.
* Prepare data: We have used *clean_data* function from the script train.py which returns features and targets as DataFrame and Series, respectively.
* Concatenated data returned from clean_data function in Dataframe, all_data.
* Split the data into train and test sets in the ratio 80% and 20% respectively.
* Save the training data in the folder training/train_data.csv.
* Get the datastore and upload train_data.csv to the datastore and create datastore referencing.
```
    datastore = ws.get_default_datastore()
    datastore.upload(src_dir='training/', target_path='data/')
    train_data = TabularDatasetFactory.from_delimited_files(path = [(datastore, ('data/train_data.csv'))])
```
* Configure *AutoMLConfig* class:
It represents the configuration for submitting an autoML experiment and contains various parameters.
```
    automl_config = AutoMLConfig(
        experiment_timeout_minutes=30,
        task='classification',
        primary_metric='accuracy',
        training_data=train_data,
        label_column_name='y',
        compute_target = compute_aml_cluster,
        n_cross_validations=5,
        iterations=45,
        max_cores_per_iteration=-1,
        max_concurrent_iterations=10)
```  
   *experiment_timeout_minutes*: Time limit in minutes for the experiment.

   *primary_metric*: Metric that you want to optimize. The best-fit model will be chosen based on this metric.

   *label_column_name*: The name of the label column whose value your model will predict.

   *n_cross_validations*: Number of cross-validation splits to perform when validation data is not specified.

   *iterations*: The total number of different algorithms and parameter combinations to test during an automated ML experiment.

   *max_cores_per_iteration*: The maximum number of threads to use for a given training iteration.-1, which means to use all the possible cores per iteration per child-run.

   *max_concurrent_iterations*: Represents the maximum number of iterations that would be executed in parallel.

* Submit AutoML experiment and show results with the RunDetails.
```
    automl_run = exp.submit(automl_config, show_output=True)
    RunDetails(automl_run).show()
```
<img src="AutoML Capture/Capture1.PNG">

* Get the best run and metric

```
    best_run, fitted_automl_best_model = automl_run.get_output()
    best_run_metrics = best_run.get_metrics()
```
Different types of algorithms supported for classification in Azure ML are:

* Logistic Regression
* Light Gradient Boosting Machine (GBM)
* Decision Tree
* Random Forest
* Naive Bayes
* Linear Support Vector Machine (SVM)
* XGBoost
* Deep Neural Network (DNN) Classifier
* Others...

## Pipeline comparison
AutoML is better in architecture than traditional machine learning model development.
* In Traditional machine learning model development requires significant domain knowledge and time to produce and compare dozens of models.
* With automated machine learning, you'll accelerate the time it takes to get production-ready ML models with great ease.

To run the hyperdrive experiment, 
* We must have a custom coded model.
* We need to define the search space and sampling method.
* We need to specify early termination policy.

To run the AutoML experiment,
* We don't need to specify all the above details in AutoML we just need to specify some parameters in *AutoMLConfig* class.
* Auto Scaling and Normalization techniques are applied by default.

### Results

Accuracy score obtained by two approaches:

* HyperDrive &ensp;: 0.91624

<img src="Hyperdrive Capture/Capture5.PNG">

For Hyperdrive best values for hyperparameter chosen are:

* C: 0.66929
* max_iter: 60

<img src="Hyperdrive Capture/Capture6.PNG">

* AutoML &ensp;&ensp;&ensp;&ensp;: 0.91567 (best model: VotingEnsemble)


<img src="AutoML Capture/Capture1.PNG">

There is no significant difference in accuracy between the two approaches. Though AutoML is a powerful tool for prediction, here Hyperdrive outperforms AutoML. 

The Accuracy of AutoML might be affected due to imbalanced data because imbalance classes were detected in input.
The algorithms used by automated ML detect imbalance when the number of samples in the minority class is equal to or fewer than 20% of the number of samples in the majority class.

<img src="AutoML Capture/Capture2.PNG">

Some algorithms are executed with accuracy.

<img src="AutoML Capture/Capture4.PNG">

Features that are impacted VotingEnsemble.

<img src="AutoML Capture/Capture5.PNG">

Some Metrics

<img src="AutoML Capture/Capture7.PNG">

### Summary Results:

* HyperDrive &ensp;: 0.91624
* AutoML &ensp;&ensp;&ensp;&ensp;: 0.91567 (best model: VotingEnsemble)

## Future work

* Accuracy is affected due to imbalanced classes in the input, because the input data is biased towards certain classes.<br>
Some of the ways you can handle imbalanced data:
    * Change performance matrix: Accuracy is not the metric to use when working with an imbalanced dataset because it is misleading. Metrics that can provide better insight include:
        * AUC_weighted
        * Confusion Matrix
        * Precision
        * Recall
        * F1 Score
    * Change sampling technique: In scikit-learn instead of using random sampling you can also grid sampling.
* For scikit learn, try a different combination of values for hyperparameter tuning.
* For AutoML, 
    * You can use more data to improve accuracy. 
    * Use AUC_weighted as primary matrix that calculates the contribution of every class based on the relative number of samples representing that class, hence is more robust against imbalance.
    * Use a weight column: automated ML supports a column of weights as input, causing rows in the data to be weighted up or down, which can be used to make a class more or less "important".


## Proof of cluster clean up

<img src="AutoML Capture/Capture10(Computer cluster delete).PNG">
