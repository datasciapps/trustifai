knowledge = """
# Data Science Task is a Core Concept
Data Preprocessing is subclass of Data Science Task
Supervised Learning is subclass of Data Science Task
Unsupervised Learning is subclass of Data Science Task
# TODO: binary vs multiclass classification
Classification is subclass of Supervised Learning
Regression is subclass of Supervised Learning
Clustering is subclass of Unsupervised Learning
Data Collection is subclass of Data Science Task
Data Loading is subclass of Data Science Task
Data Cleaning is subclass of Data Preprocessing
Data Augmentation is subclass of Data Preprocessing
Data Integration is subclass of Data Preprocessing
Data Exploration is subclass of Data Science Task
Data Visualization is subclass of Data Science Task
Statistical Analysis is subclass of Data Science Task
Natural Language Processing is subclass of Data Science Task
Image Processing is subclass of Data Science Task
Video Processing is subclass of Data Science Task
Time Series Analysis is subclass of Data Science Task
Anomaly Detection is subclass of Data Science Task
Fraud Detection is subclass of Anomaly Detection
Network Intrusion Detection is subclass of Anomaly Detection
Model Selection is subclass of Data Science Task
Model Evaluation is subclass of Data Science Task
Model Deployment is subclass of Data Science Task
Data Ingestion is subclass of Data Collection
Data Scraping is subclass of Data Collection
Data Warehousing is subclass of Data Collection
Batch Data Loading is subclass of Data Loading
Streaming Data Loading is subclass of Data Loading
Missing Data Imputation is subclass of Data Cleaning
Data Transformation is subclass of Data Preprocessing
Data Normalization is subclass of Data Transformation
Data Standardization is subclass of Data Transformation
Feature Engineering is subclass of Data Preprocessing
Feature Selection is subclass of Feature Engineering
Feature Extraction is subclass of Feature Engineering
Exploratory Data Analysis is subclass of Data Exploration
Data Profiling is subclass of Data Exploration
Interactive Visualization is subclass of Data Visualization
Dashboarding is subclass of Data Visualization
Hypothesis Testing is subclass of Statistical Analysis
Statistical Modeling is subclass of Statistical Analysis
Bayesian Inference is subclass of Statistical Analysis
Binary Classification is subclass of Classification
Multiclass Classification is subclass of Classification
Multi-label Classification is subclass of Classification
Linear Regression is subclass of Regression
Polynomial Regression is subclass of Regression
Logistic Regression is subclass of Regression
Time Series Forecasting is subclass of Regression
K-Means Clustering is subclass of Clustering
Hierarchical Clustering is subclass of Clustering
DBSCAN is subclass of Clustering
Dimensionality Reduction is subclass of Unsupervised Learning
Principal Component Analysis is subclass of Dimensionality Reduction
t-SNE is subclass of Dimensionality Reduction
Autoencoder-based Reduction is subclass of Dimensionality Reduction
Association Rule Mining is subclass of Unsupervised Learning
Outlier Detection is subclass of Anomaly Detection
Cross-Validation is subclass of Model Selection
Hyperparameter Tuning is subclass of Model Selection
Ensemble Methods is subclass of Model Selection
Confusion Matrix Analysis is subclass of Model Evaluation
ROC Analysis is subclass of Model Evaluation
Precision and Recall Analysis is subclass of Model Evaluation
Model Serving is subclass of Model Deployment
Model Monitoring is subclass of Model Deployment
Model Versioning is subclass of Model Deployment
A/B Testing is subclass of Model Deployment
Text Mining is subclass of Natural Language Processing
Sentiment Analysis is subclass of Natural Language Processing
Named Entity Recognition is subclass of Natural Language Processing
Topic Modeling is subclass of Natural Language Processing
Machine Translation is subclass of Natural Language Processing
Computer Vision is subclass of Image Processing
Object Detection is subclass of Computer Vision
Image Segmentation is subclass of Computer Vision
Facial Recognition is subclass of Computer Vision
Video Analytics is subclass of Video Processing
Action Recognition is subclass of Video Processing
Video Summarization is subclass of Video Processing
Trend Analysis is subclass of Time Series Analysis
Seasonality Analysis is subclass of Time Series Analysis
Change Point Detection is subclass of Time Series Analysis

#Data Science Task is a Core Concept
Data Privacy, Ethics & Fairness is subclass of Data Science Task
Bias Detection & Fairness Evaluation is subclass of Data Privacy, Ethics & Fairness
Data Anonymization & Privacy Preservation is subclass of Data Privacy, Ethics & Fairness
Ethical Review & Compliance is subclass of Data Privacy, Ethics & Fairness
Infrastructure & Automation is subclass of Data Science Task
ETL & Data Pipeline Orchestration is subclass of Infrastructure & Automation
Cloud Infrastructure & Big Data Processing is subclass of Infrastructure & Automation
CI/CD for Data Science Projects is subclass of Infrastructure & Automation
TrainTestSplit is subclass of Model Selection
KFold is subclass of Model Selection
StratifiedKFold is subclass of Model Selection
GroupShuffleSplit is subclass of Model Selection
KFold is subclass of Model Selection
LeaveOneGroupOut is subclass of Model Selection
LeaveOneOut is subclass of Model Selection
LeavePGroupsOut is subclass of Model Selection
LeavePOut is subclass of Model Selection
PredefinedSplit is subclass of Model Selection
RepeatedKFold is subclass of Model Selection
RepeatedStratifiedKFold is subclass of Model Selection
ShuffleSplit is subclass of Model Selection
StratifiedGroupKFold is subclass of Model Selection
StratifiedKFold is subclass of Model Selection
StratifiedShuffleSplit is subclass of Model Selection
TimeSeriesSplit is subclass of Model Selection
check_cv is subclass of Model Selection
train_test_split is subclass of Model Selection
GridSearchCV is subclass of Model Selection
HalvingGridSearchCV is subclass of Model Selection
HalvingRandomSearchCV is subclass of Model Selection
ParameterGrid is subclass of Model Selection
ParameterSampler is subclass of Model Selection
RandomizedSearchCV is subclass of Model Selection
FixedThresholdClassifier is subclass of Model Selection
TunedThresholdClassifierCV is subclass of Model Selection
cross_val_predict is subclass of Model Selection
cross_val_score is subclass of Model Selection
cross_validate is subclass of Model Selection
learning_curve is subclass of Model Selection
permutation_test_score is subclass of Model Selection
validation_curve is subclass of Model Selection
LearningCurveDisplay is subclass of Model Selection
ValidationCurveDisplay is subclass of Model Selection
Data Normalization is subclass of Data Preprocessing
MinMaxScaler is subclass of Data Normalization
StandardScaler is subclass of Data Normalization
Data Encoding is subclass of Data Preprocessing
OneHotEncoder is subclass of Data Encoding
SVM is subclass of Classification
Predictive Performance is a Measure
Accuracy is a Metric
Recall is a Metric
ROC AUC Score is a Metric
Accuracy contributes to Predictive Performance
Recall contributes to Predictive Performance
ROC AUC Score contributes to Predictive Performance
# Evaluation Procedure is a Core Concept
# TODO: 'conditions' vs 'preferences'
# E.g., for smaller datasets, use cross-validation or LOOCV to make the best use of limited data
# Interface is a Core Concept
Sklearn Estimator is an Interface
Sklearn Estimator calls __init__; calls fit; calls predict
Sklearn Transformer is an Interface
Sklearn Transformer calls __init__; calls fit; calls transform
Function is an Interface
TrainTestSplit might introduce Selection Bias
TrainTestSplit might introduce Sampling Bias
TrainTestSplit might introduce Data Leakage
TrainTestSplit might introduce Temporal Bias
TrainTestSplit might introduce Class Imbalance
TrainTestSplit might introduce Group Bias
TrainTestSplit might introduce Covariate Shift
# TODO: libraries and versions
Requirement is a Core Concept
# https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai
# TODO: Add 'has context' with descriptions and definitions
Fairness is a Requirement
Lawfulness is a Requirement
Human Agency is a Requirement
Human Oversight is a Requirement
Safety is a Requirement
Technical Robustness is a Requirement
Privacy is a Requirement
Data Governance is a Requirement
Transparency is a Requirement
Diversity is a Requirement
Non-discrimination is a Requirement
Societal Well-being is a Requirement
Environmental Well-being is a Requirement
Accountability is a Requirement
Acceptance is a Requirement
#Concepts of Fairness
Equality is subclass of Fairness
Equity is subclass of Fairness
Transparency is subclass of Fairness
Confidentiality is subclass of Fairness
Voice is subclass of Fairness
Timeliness is subclass of Fairness
Impartiality is subclass of Fairness
Rationality is subclass of Fairness
Accountability is subclass of Fairness
Flexibility is subclass of Fairness
Dignity is subclass of Fairness
Bias is threat to Fairness
Equality is dimension of Fairness
Equity is dimension of Fairness
#Measures of fairness
Fairness is a Measure
Demographic Parity is synonym of Statistical Parity
Equalized Odds contributes to Fairness
Equal Opportunity contributes to Fairness
Error-Rate Parity contributes to Fairness
Individual Fairness contributes to Fairness
Distance-based Fairness is synonym of Individual Fairness
Counterfactual Fairness contributes to Fairness
Calibration Fairness contributes to Fairness
Equal mis-opportunity is synonym of Predictive Equality
Average odds contributes to Fairness
Chi-Square Fairness Test contributes to Fairness
Zemel’s Fair Clustering Metric contributes to Fairness
Group Fairness contributes to Fairness
Group Fairness is a Measure
Parity-based Fairness is a Measure
Parity-based Fairness is subclass of Group Fairness
Parity-based Fairness contributes to Fairness
Demographic Parity contributes to Parity-based Fairness
Conditional Demographic Disparity contributes to Parity-based Fairness
Disparate Impact contributes to Parity-based Fairness
Statistical Parity contributes to Parity-based Fairness
Statistical Parity Difference contributes to Parity-based Fairness
Demographic Parity is a Metric
Conditional Demographic Disparity is a Metric
Disparate Impact is a Metric
Statistical Parity is a Metric
Statistical Parity Difference is a Metric
Calibration-based Fairness is a Measure
Calibration-based Fairness is subclass of Group Fairness
Calibration-based Fairness contributes to Fairness
Test Fairness contributes to Calibration-based Fairness
Calibration is synonym of Test Fairness
Matching Conditional Frequencies is synonym of Test Fairness
Well Calibration contributes to Calibration-based Fairness
Test Fairness is a Metric
Well Calibration is a Metric
Score-based Fairness is a Measure
Score-based Fairness is subclass of Group Fairness
Score-based Fairness contributes to Fairness
Balance for Positive Class contributes to Score-based Fairness
Balance for Negative Class contributes to Score-based Fairness
Bayesian Fairness contributes to Score-based Fairness
Balance for Positive Class is a Metric
Balance for Negative Class is a Metric
Bayesian Fairness is a Metric
Error-based Fairness is a Measure
Error-based Fairness is subclass of Group Fairness
Error-based Fairness contributes to Fairness
Equalized Odds Difference contributes to Error-based Fairness
Generalized Equalized Odds Difference contributes to Error-based Fairness
Equal Opportunity contributes to Error-based Fairness
Overall Accuracy Equality contributes to Error-based Fairness
Conditonal use accuracy equality contributes to Error-based Fairness
Treatment euqality contributes to Error-based Fairness
Equalizing disincentives contributes to Error-based Fairness
Conditional equal opportunity contributes to Error-based Fairness
Predictive Parity contributes to Error-based Fairness
Error Rate Difference contributes to Error-based Fairness
Error Rate Parity is synonym of Error Rate Difference
Error Rate Ratio contributes to Error-based Fairness
Error Rate Difference Ratio contributes to Error-based Fairness
Accuracy Parity contributes to Error-based Fairness
Balanced Group Fairness contributes to Error-based Fairness
Average Odds Difference contributes to Error-based Fairness
Average Absolute Odds Difference contributes to Error-based Fairness
Average Predictve Value Difference contributes to Error-based Fairness
False Negative Rate Difference contributes to Error-based Fairness
False Positive Rate Difference contributes to Error-based Fairness
False Positive Rate Ratio contributes to Error-based Fairness
False Negative Rate Ratio contributes to Error-based Fairness
True Positive Rate Difference contributes to Error-based Fairness
False Ommission Rate Ratio contributes to Error-based Fairness
FNR Difference is synonym of False Negative Rate Difference
FPR Difference is synonym of False Positive Rate Difference
TPR Difference is synonym of True Positive Rate Difference
Balanced Accuracy Difference contributes to Error-based Fairness
False Discovery Rate Difference contributes to Error-based Fairness
False Omission Rate Difference contributes to Error-based Fairness
Balanced Error Rate Difference contributes to Error-based Fairness
Equalized Odds Difference is a Metric
Generalized Equalized Odds Difference is a Metric
Average Predictve Value Difference is a Metric
Equal Opportunity Difference is a Metric
Overall Accuracy Equality is a Metric
Conditonal use accuracy equality is a Metric
Treatment euqality is a Metric
Equalizing disincentives is a Metric
Conditional equal opportunity is a Metric
Predictive Parity is a Metric
Error Rate Difference is a Metric
Error Rate Ratio is a Metric
Accuracy Parity is a Metric
Balanced Group Fairness is a Metric
Average Odds Difference is a Metric
Average Absolute Odds Difference is a Metric
False Negative Rate Difference is a Metric
False Positive Rate Difference is a Metric
False Positive Rate Ratio is a Metric
True Positive Rate Difference is a Metric
False Negative Rate Ratio is a Metric
False Ommission Rate Ratio is a Metric
Balanced Accuracy Difference is a Metric
False Discovery Rate Difference is a Metric
False Omission Rate Difference is a Metric
Balanced Error Rate Difference is a Metric
Other Group Fairness is subclass of Group Fairness
Other Group Fairness contributes to Fairness
Predictive Equality contributes to Other Group Fairness
Group Fairness contributes to Other Group Fairness
Predictive Equality is a Metric
Group Fairness is a Metric
Distributional Fairness is subclass of Fairness
Distributional Fairness contributes to Fairness
Theil Index contributes to Distributional Fairness
Wasserstein Distance contributes to Distributional Fairness
Jensen-Shannon Divergence Fairness contributes to Distributional Fairness
Kolmogorov-Smirnov Fairness Distance contributes to Distributional Fairness
Kullback-Leibler Divergence Fairness contributes to Distributional Fairness
Shannon Entropy Fairness contributes to Distributional Fairness
Bayesian Fairness Criterion contributes to Distributional Fairness
Mutual Information Fairness contributes to Distributional Fairness
Generalized Entropy Index contributes to Distributional Fairness
Theil Index is a subclass of Generalized Entropy Index
Coefficient of Variation contributes to Distributional Fairness
Coefficient of Variation is subclass of Generalized Entropy Index
Between Group Generalized Entropy Index is subclass of Generalized Entropy Index
Between Group Coefficient of Variation is subclass of Between Group Generalized Entropy Index
Between Group Theil Index is subclass of Between Group Generalized Entropy Index
Between All Groups Generalized Entropy Index is subclass of Generalized Entropy Index
Between All Groups Coefficient of Variation is subclass of Between All Groups Generalized Entropy Index
Between All Groups Theil Index is subclass of Between All Groups Generalized Entropy Index

Theil Index is a Metric
Wasserstein Distance is a Metric
Jensen-Shannon Divergence Fairness is a Metric
Kolmogorov-Smirnov Fairness Distance is a Metric
Kullback-Leibler Divergence Fairness is a Metric
Shannon Entropy Fairness is a Metric
Bayesian Fairness Criterion is a Metric
Mutual Information Fairness is a Metric
Generalized Entropy Index is a Metric
Coefficient of Variation is a Metric
Between Group Generalized Entropy Index is a Metric
Between Group Coefficient of Variation is a Metric
Between Group Theil Index is a Metric
Between All Groups Generalized Entropy Index is a Metric
Between All Groups Coefficient of Variation is a Metric
Between All Groups Theil Index is a Metric

Individual Fairness is subclass of Fairness
Individual Equalized Odds contributes to Individual Fairness
Fairness Lipschitz Condition contributes to Individual Fairness
Fair k-Nearest Neighbors contributes to Individual Fairness
Consistency Score contributes to Individual Fairness
Individual Equalized Odds is a Metric
Fairness Lipschitz Condition is a Metric
Fair k-Nearest Neighbors is a Metric
Consistency Score is a Metric
Causal and Counterfactual Fairness is subclass of Fairness
Causal and Counterfactual Fairness contributes to Fairness
Counterfactual Fairness contributes to Causal and Counterfactual Fairness
Causal Fairness contributes to Causal and Counterfactual Fairness
Path-Specific Fairness contributes to Causal and Counterfactual Fairness
Total Effect Fairness contributes to Causal and Counterfactual Fairness
Direct and Indirect Effect Fairness contributes to Causal and Counterfactual Fairness
Shapley Fairness Index contributes to Causal and Counterfactual Fairness
Counterfactual Fairness is a Metric
Causal Fairness is a Metric
Path-Specific Fairness is a Metric
Total Effect Fairness is a Metric
Direct and Indirect Effect Fairness is a Metric
Shapley Fairness Index is a Metric
Fairness through Awareness contributes to Fairness
Fairness through Unawareness contributes to Fairness
Ranking Fairness is subclass of Fairness
Ranking Fairness contributes to Fairness
Discounted Cumulative Fairness contributes to Ranking Fairness
Rank Parity Fairness contributes to Ranking Fairness
Fairness-aware Mean Average Precision contributes to Ranking Fairness
Discounted Cumulative Fairness is a Metric
Rank Parity Fairness is a Metric
Fairness-aware Mean Average Precision is a Metric
#Metrics of Fairness from Ko's lit review
True Positive Rate Parity is synonym of Equal Opportunity Difference
TPR Parity is synonym of True Positive Rate Parity
False Positive Rate Parity is a Metric
False Negative Rate Parity is a Metric
Specificity Parity is a Metric
TNR Parity is synonym of Specificity Parity
PPV Parity is synonym of Predictive Parity
Negative Predictive Value Parity is a Metric
Matthews Correlation Coefficient Parity is a Metric
AUROC Parity is a Metric
Calibration Parity is a Metric
ECE Parity is synonym of Calibration Parity
Brier Score Parity is a Metric
Treatment Equality is a Metric
Conditional Demographic Parity is a Metric
Group Benefit Equality is a Metric
Individual Fairness is a Metric
Consistency is a Metric
Counterfactual Token Fairness is a Metric
#TODO: classify Smoothed Empirical Differential Fairness better
Smoothed Empirical Differential Fairness is a Metric
Equalized Odds contributes to Fairness
False Positive Rate Parity contributes to Fairness
False Negative Rate Parity contributes to Fairness
Specificity Parity contributes to Fairness
Negative Predictive Value Parity contributes to Fairness
Matthews Correlation Coefficient Parity contributes to Fairness
AUROC Parity contributes to Fairness
Calibration Parity contributes to Fairness
Brier Score Parity contributes to Fairness
Treatment Equality contributes to Fairness
Conditional Demographic Parity contributes to Fairness
Group Benefit Equality contributes to Fairness
Consistency contributes to Fairness
Counterfactual Token Fairness contributes to Fairness
#Bias
Bias is a Risk
Algorithmic Bias is subclass of Bias
Demographic Bias is subclass of Bias
Popularity Bias is subclass of Bias
Homophily is subclass of Bias
Nationality Bias is subclass of Bias
Self-selection Bias is subclass of Bias
Underlying Distribution Skew is subclass of Bias
Historical Bias is subclass of Bias
Sampling Bias is subclass of Bias
Data Leakage might introduce Bias
Temporal Bias is subclass of Bias
Class Imbalance is subclass of Bias
Data Bias is subclass of Bias
Group Bias is subclass of Bias
Stratification Bias is synonym of Group Bias
Covariate Shift might introduce Bias
Representation Bias is subclass of Bias
Measurement Bias is subclass of Bias
Omitted variable Bias is subclass of Bias
Evaluation Bias is subclass of Bias
Aggregation Bias is subclass of Bias
User interaction bias is subclass of Bias
Population Bias is subclass of Bias
Deployment Bias is subclass of Bias
Feedback Loop contributes to Bias
Unconscious Bias is subclass of Bias
Cognitive Bias is subclass of Bias
Confirmation Bias is subclass of Bias
Selection Bias is subclass of Bias
Reporting Bias is subclass of Bias
Person is an Entity
First Name attributes to Person
Middle Name attributes to Person
Last Name attributes to Person
Date of Birth attributes to Person
Age attributes to Person
Gender attributes to Person
Sex attributes to Person
Race attributes to Person
Ethnicity attributes to Person
Disability status attributes to Person
Religion attributes to Person
Sexual orientation attributes to Person
National origin attributes to Person
Marital status attributes to Person
Socioeconomic status attributes to Person
Gender identity attributes to Person
Gender expression attributes to Person
Caste attributes to Person
Nationality attributes to Person
Physical health status attributes to Person
Mental health status attributes to Person
Neurodiversity status attributes to Person
Chronic illness status attributes to Person
Pregnancy status attributes to Person
Parental status attributes to Person
Family structure attributes to Person
Employment status attributes to Person
Income level attributes to Person
Education level attributes to Person
Literacy level attributes to Person
Genetic information attributes to Person
Biometric data attributes to Person
Physical appearance attributes to Person
Place of birth attributes to Person
Language spoken attributes to Person
Accent attributes to Person
Dialect attributes to Person
Cultural background attributes to Person
Indigenous status attributes to Person
Political affiliation attributes to Person
Philosophical beliefs attributes to Person
Union membership attributes to Person
Digital access attributes to Person
AI literacy attributes to Person
#protected attributes
Gender is a Protected Attribute
Race is a Protected Attribute
Sexual orientation is a Protected Attribute
Religion is a Protected Attribute
Ethnicity is a Protected Attribute
Sex is a Protected Attribute
Disability status is a Protected Attribute
National origin is a Protected Attribute
Age is a Protected Attribute
Gender identity is a Protected Attribute
Gender expression is a Protected Attribute
Caste is a Protected Attribute
Nationality is a Protected Attribute
Physical health status is a Protected Attribute
Mental health status is a Protected Attribute
Neurodiversity status is a Protected Attribute
Chronic illness status is a Protected Attribute
Pregnancy status is a Protected Attribute
Marital status is a Protected Attribute
Parental status is a Protected Attribute
Family structure is a Protected Attribute
Socioeconomic status is a Protected Attribute
Employment status is a Protected Attribute
Income level is a Protected Attribute
Education level is a Protected Attribute
Literacy level is a Protected Attribute
Genetic information is a Protected Attribute
Biometric data is a Protected Attribute
Physical appearance is a Protected Attribute
Place of birth is a Protected Attribute
Language spoken is a Protected Attribute
Accent is a Protected Attribute
Dialect is a Protected Attribute
Cultural background is a Protected Attribute
Indigenous status is a Protected Attribute
Political affiliation is a Protected Attribute
Philosophical beliefs is a Protected Attribute
Union membership is a Protected Attribute
Digital access is a Protected Attribute
AI literacy is a Protected Attribute
Classification should ensure Fairness
Tabular Data is subclass of Data
# Metrics, e.g., https://www.kaggle.com/code/alexisbcook/ai-fairness
Demographic Parity is subclass of Equality
Demographic Parity contributes to Equality
Equality of Opportunity  is subclass of Equality
Equality of Opportunity contributes to Equality
Equal Accuracy is subclass of Equality
Equal Accuracy contributes to Equality
Group Unaware is subclass of Equality
Group Unaware contributes to Equality
# https://github.com/understandable-machine-intelligence-lab/Quantus
Explainability is a Requirement
Explanation Robustness is subclass of Explainability
Explanation Consistency is a Measure
Quantus_Consistency is a Metric
Explanation Consistency contributes to Explanation Robustness
Quantus_Consistency implements Explanation Consistency
Quantus_Consistency applies to Tabular Data
# X implements Demographic parity
Immediate Alternatives is a Mitigation Action
Immediate Alternatives might mitigate Bias
KFold might mitigate Selection Bias
ShuffleSplit might mitigate Selection Bias
StratifiedKFold might mitigate Sampling Bias
StratifiedShuffleSplit might mitigate Sampling Bias
PredefinedSplit might mitigate Data Leakage
TimeSeriesSplit might mitigate Data Leakage
TimeSeriesSplit might mitigate Temporal Bias
StratifiedKFold might mitigate Class Imbalance
StratifiedShuffleSplit might mitigate Class Imbalance
GroupKFold might mitigate Group Bias
LeaveOneGroupOut might mitigate Group Bias
InstanceReweighting might mitigate Covariate Shift
#TODO: make the taxonomy clearer with the below

Data Sampling might mitigate Bias

Pre-processing Method is a Mitigation Action
In-processing Method is a Mitigation Action
Post-processing Method is a Mitigation Action
Adversarial Learning is a Pre-processing Method
CFGAN is subclass of Adversarial Learning
FAIR is subclass of Adversarial Learning
FairGAN is subclass of Adversarial Learning
FUR is subclass of Adversarial Learning
Fair Transfer Learning is subclass of Adversarial Learning
Fair Representations is subclass of Adversarial Learning
One Network is subclass of Adversarial Learning
Blinding is a Pre-processing Method
Blinding might mitigate Bias
Fairness through Unawareness is synonym of Blinding
Ommision is synonym of Blinding
Immunity is subclass of Blinding
Partial Blinding is synonym of Blinding
Causal Methods is a Pre-processing Method  
Causal Bayesian Networks is subclass of Causal Methods
Counterfactual Fairness is subclass of Causal Methods
Learn Stochastic Decision Policies is subclass of Causal Methods
Themis is subclass of Causal Methods
Causal Database Repair is subclass of Causal Methods
Competing Causal Explanations is subclass of Causal Methods
Sampling is a Pre-processing Method
Subgroup Analysis is a Pre-processing Method
Transformation is a Pre-processing Method
Transformation is a Post-processing Method
Relabelling is subclass of Transformation
Data Massaging is synonym of Relabelling
Perturbation is subclass of Transformation
Reweighing is a Pre-processing Method
Reweighing is a In-processing Method
Regularization is a In-processing Method
Constraint Optimization is a In-processing Method
Adversarial Learning is a In-processing Method
Bandits is a In-processing Method
Calibration is a Post-processing Method
Thresholding is a Post-processing Method

sklearn.model_selection.train_test_split is an Operator
sklearn.model_selection.train_test_split implements TrainTestSplit
sklearn.model_selection.train_test_split is a Function
sklearn.model_selection.KFold is an Operator
sklearn.model_selection.KFold implements KFold
sklearn.model_selection.KFold is a Function
sklearn.model_selection.StratifiedKFold is an Operator
sklearn.model_selection.StratifiedKFold implements StratifiedKFold
sklearn.model_selection.StratifiedKFold is a Function
sklearn.preprocessing.MinMaxScaler is an Operator
sklearn.preprocessing.MinMaxScaler is a Sklearn Transformer
sklearn.preprocessing.MinMaxScaler implements MinMaxScaler
sklearn.preprocessing.StandardScaler is an Operator
sklearn.preprocessing.StandardScaler is a Sklearn Transformer
sklearn.preprocessing.StandardScaler implements StandardScaler
sklearn.preprocessing.OneHotEncoder is an Operator
sklearn.preprocessing.OneHotEncoder is a Sklearn Transformer
sklearn.preprocessing.OneHotEncoder implements OneHotEncoder
sklearn.svm.SVC is an Operator
sklearn.svm.SVC is a Sklearn Estimator
sklearn.svm.SVC implements SVM

#fairness toolkits and versions
aif360.algorithms.Transformer is an Interface
aif360.algorithms.Transformer calls __init__; calls fit; calls transform
# aif360.algorithms.Transformer calls __init__; fit_predict; calls x
# aif360.algorithms.Transformer calls __init__; predict; calls x
# aif360.algorithms.Transformer calls __init__; fit_transform; calls x
ARTClassifier is a In-Procesing Method
AdversarialDebiasing is a In-Procesing Method
ExponentiatedGradientReduction is a In-Procesing Method
GerryFairClassifier is a In-Procesing Method
GridSearchReduction is a In-Procesing Method
MetaFairClassifier is a In-Procesing Method
PrejudiceRemover is a In-Procesing Method
CalibratedEqOddsPostprocessing is a Post-Procesing Method
EqOddsPostprocessing is a Post-Procesing Method
RejectOptionClassification is a Post-Processing Method
DeterministicReranking is a Post-Processing Method
DisparateImpactRemover is a Pre-Processing Method
Reweighing is a Pre-Processing Method
LFR is a Pre-Processing Method
OptimPreproc is a Pre-Processing Method
aif360.algorithms.inprocessing.AdversarialDebiasing is an Operator
aif360.algorithms.inprocessing.AdversarialDebiasing is a aif360.algorithms.Transformer
aif360.algorithms.inprocessing.AdversarialDebiasing implements AdversarialDebiasing
aif360.algorithms.inprocessing.ARTClassifier is an Operator
aif360.algorithms.inprocessing.ARTClassifier is a aif360.algorithms.Transformer
aif360.algorithms.inprocessing.ARTClassifier implements ARTClassifier
aif360.algorithms.inprocessing.ExponentiatedGradientReduction is an Operator
aif360.algorithms.inprocessing.ExponentiatedGradientReduction is a aif360.algorithms.Transformer
aif360.algorithms.inprocessing.ExponentiatedGradientReduction implements ExponentiatedGradientReduction
aif360.algorithms.inprocessing.GerryFairClassifier is an Operator
aif360.algorithms.inprocessing.GerryFairClassifier is a aif360.algorithms.Transformer
aif360.algorithms.inprocessing.GerryFairClassifier implements GerryFairClassifier
aif360.algorithms.inprocessing.PrejudiceRemover is an Operator
aif360.algorithms.inprocessing.PrejudiceRemover is a aif360.algorithms.Transformer
aif360.algorithms.inprocessing.PrejudiceRemover implements PrejudiceRemover
aif360.algorithms.inprocessing.GridSearchReduction is an Operator
aif360.algorithms.inprocessing.GridSearchReduction is a aif360.algorithms.Transformer
aif360.algorithms.inprocessing.GridSearchReduction implements GridSearchReduction
aif360.algorithms.inprocessing.MetaFairClassifier is an Operator
aif360.algorithms.inprocessing.MetaFairClassifier is a aif360.algorithms.Transformer
aif360.algorithms.inprocessing.MetaFairClassifier implements MetaFairClassifier
aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing is an Operator
aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing is a aif360.algorithms.Transformer
aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing implements CalibratedEqOddsPostprocessing
aif360.algorithms.postprocessing.EqOddsPostprocessing is an Operator
aif360.algorithms.postprocessing.EqOddsPostprocessing is a aif360.algorithms.Transformer
aif360.algorithms.postprocessing.EqOddsPostprocessing implements EqOddsPostprocessing
aif360.algorithms.postprocessing.RejectOptionClassification is an Operator
aif360.algorithms.postprocessing.RejectOptionClassification is a aif360.algorithms.Transformer
aif360.algorithms.postprocessing.RejectOptionClassification implements RejectOptionClassification
aif360.algorithms.preprocessing.DisparateImpactRemover is an Operator
aif360.algorithms.preprocessing.DisparateImpactRemover is a ai360.algorithms.Transformer
aif360.algorithms.preprocessing.DisparateImpactRemover implements DisparateImpactRemover
aif360.algorithms.preprocessing.LFR is an Operator
aif360.algorithms.preprocessing.LFR is a aif360.algorithms.Transformer
aif360.algorithms.preprocessing.LFR implements LFR
aif360.algorithms.preprocessing.OptimPreproc is an Operator
aif360.algorithms.preprocessing.OptimPreproc is a aif360.algorithms.Transformer
aif360.algorithms.preprocessing.OptimPreproc implements OptimPreproc
aif360.algorithms.preprocessing.Reweighing is an Operator
aif360.algorithms.preprocessing.Reweighing is a aif360.algorithms.Transformer
aif360.algorithms.preprocessing.Reweighing implements Reweighing

aif360.datasets.DatasetMetric is an Interface
aif360.datasets.BinaryLabelDatasetMetric is a aif360.datasets.DatasetMetric
aif360.datasets.BinaryLabelDatasetMetric.consistency implements Consistency Score
aif360.datasets.BinaryLabelDatasetMetric.disparate_impact implements Disparate Impact
aif360.datasets.BinaryLabelDatasetMetric.mean_difference implements Statistical Parity Difference
aif360.datasets.BinaryLabelDatasetMetric.statistical_parity_difference implements Statistical Parity Difference
aif360.datasets.BinaryLabelDatasetMetric.smoothed_empirical_differential_fairness implements Equal Opportunity Difference
aif360.datasets.ClassificationMetric is a aif360.datasets.BinaryLabelDatasetMetric
aif360.datasets.ClassificationMetric.false_positive_rate_difference implements False Positive Rate Difference
aif360.datasets.ClassificationMetric.false_negative_rate_difference implements False Negative Rate Difference
aif360.datasets.ClassificationMetric.false_discovery_rate_difference implements False Discovery Rate Difference
aif360.datasets.ClassificationMetric.false_omission_rate_difference implements False Omission Rate Difference
aif360.datasets.ClassificationMetric.false_positive_rate_ratio implements False Positive Rate Ratio
aif360.datasets.ClassificationMetric.false_negative_rate_ratio implements False Negative Rate Ratio
aif360.datasets.ClassificationMetric.false_omission_rate_ratio implements False Omission Rate Ratio
aif360.datasets.ClassificationMetric.error_rate_difference implements Error Rate Difference
aif360.datasets.ClassificationMetric.error_rate_ratio implements Error Rate Ratio
aif360.datasets.ClassificationMetric.equal_opportunity_difference implements Equal Opportunity Difference
aif360.datasets.ClassificationMetric.generalized_entropy_index implements Generalized Entropy Index
aif360.datasets.ClassificationMetric.true_positive_rate_difference implements True Positive Rate Difference
aif360.datasets.ClassificationMetric.theil_index implements Theil Index
aif360.datasets.ClassificationMetric.equalized_odds_difference implements Equalized Odds Difference
aif360.datasets.ClassificationMetric.generalized_equalized_odds_difference implements Generalized Equalized Odds Difference
aif360.datasets.ClassificationMetric.coefficient_of_variation implements Coefficient of Variation
aif360.datasets.ClassificationMetric.between_all_groups_coefficient_of_variation implements Between All Groups Coefficient of Variation
aif360.datasets.ClassificationMetric.between_all_groups_generalized_entropy_index implements Between All Groups Generalized Entropy Index
aif360.datasets.ClassificationMetric.between_all_groups_theil_index implements Between All Groups Theil Index
aif360.datasets.ClassificationMetric.between_group_coefficient_of_variation implements Between Group Coefficient of Variation
aif360.datasets.ClassificationMetric.between_group_generalized_entropy_index implements Between Group Generalized Entropy Index
aif360.datasets.ClassificationMetric.between_group_theil_index implements Between Group Theil Index
aif360.datasets.ClassificationMetric.false_positive_rate_difference is an Operator
aif360.datasets.ClassificationMetric.false_negative_rate_difference is an Operator
aif360.datasets.ClassificationMetric.false_discovery_rate_difference is an Operator
aif360.datasets.ClassificationMetric.false_omission_rate_difference is an Operator
aif360.datasets.ClassificationMetric.false_positive_rate_ratio is an Operator
aif360.datasets.ClassificationMetric.false_negative_rate_ratio is an Operator
aif360.datasets.ClassificationMetric.false_omission_rate_ratio is an Operator
aif360.datasets.ClassificationMetric.error_rate_difference is an Operator
aif360.datasets.ClassificationMetric.error_rate_ratio is an Operator
aif360.datasets.ClassificationMetric.equal_opportunity_difference is an Operator
aif360.datasets.ClassificationMetric.generalized_entropy_index is an Operator
aif360.datasets.ClassificationMetric.true_positive_rate_difference is an Operator
aif360.datasets.ClassificationMetric.theil_index is an Operator
aif360.datasets.ClassificationMetric.equalized_odds_difference is an Operator
aif360.datasets.ClassificationMetric.generalized_equalized_odds_difference is an Operator
aif360.datasets.ClassificationMetric.coefficient_of_variation is an Operator
aif360.datasets.ClassificationMetric.between_all_groups_coefficient_of_variation is an Operator
aif360.datasets.ClassificationMetric.between_all_groups_generalized_entropy_index is an Operator
aif360.datasets.ClassificationMetric.between_all_groups_theil_index is an Operator
aif360.datasets.ClassificationMetric.between_group_coefficient_of_variation is an Operator
aif360.datasets.ClassificationMetric.between_group_generalized_entropy_index is an Operator
aif360.datasets.ClassificationMetric.between_group_theil_index is an Operator
aif360.datasets.BinaryLabelDatasetMetric.consistency is an Operator
aif360.datasets.BinaryLabelDatasetMetric.disparate_impact is an Operator
aif360.datasets.BinaryLabelDatasetMetric.mean_difference is an Operator
aif360.datasets.BinaryLabelDatasetMetric.statistical_parity_difference is an Operator
aif360.datasets.BinaryLabelDatasetMetric.smoothed_empirical_differential_fairness is an Operator






#maybe add information about the type of Dataset the above operators take
"""
