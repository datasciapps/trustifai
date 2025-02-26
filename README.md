# trustifai

a template for the config file `/config.yaml`
```yaml
neo4j:
  uri:
  username:
  password:
  dbname:
```

## Research on Fairness

Much of the literature emphasiyes binary classification. 
Approaches in ML have "found dark skin unattractive", claim that "black people offend more", and created a neo-nazi sexbot. 

In the first wave of research in fairness. 

Fairness literature focuses on: 
1. technical aspects of bias and fairness in ML
	1. Typically pplied prior to modeling (pre-processing), at the point of modeling (in-processing), or after modeling (post-processing), that is, they emphasize **intervention**. 
2. or theorizes on the social, legal, and ethical aspects of ML discrimination. 
3. ![[Screenshot 2025-02-25 at 17.33.01.png]]


## Key methodological components

1. **Sensitive and protected variables and (Un)privileged groups** 
	1. Most approaches to mitigate bias, discrimination, unfairness are based on the notion of protected/sensitive variables and on (un)privileged groups
		1. (Un)privileged groups: defined by one or more sensitive variables that are disproportionately less or more likey to be positively classified. 
	2. **Protected variables**: define the aspects of data that are socioculturally precarious for the application of ML. Basically any other feature of the data that involves or concerns people. 

"""
Not considering correlated sensitive variables has been shown to increase the risk of discrimination. [54, 88, 89, 89, 196, 202, 231, 247, 286, 308] from the Fairness survey. Fairness literature often overlooks the effects of correlated sensitive variables  on fairness. 
"""

2. **Metrics**
	1. Usually emphasize individual or group fairness
		1. Group fairness is further differentiated to within group and between group fairness. 
3. **Technical fairness interventions: Pre-, In-, and Post-processing approaches**
	1. **pre-processing**: approaches recognize that ofter an issues is the data itself, and the distributions of specific sensitive or protected variables are biased, discriminatory and/or imbalanced. Thus, preprocessing approaches tend to alter the sample distributions of protected variables or more generally perform specific transformations on the data with the aim to remove discrimination from the training data. The main idea here is to train a model on a “**repaired**” dataset.
	2. **In-processing**: approaches recognize that modeling techniques often become biased by dominant features, other distributional effects, or try to find a balance between multiple model objectives, for example, having a model that is both accurate and fair. In-processing approaches tackle this by often incorporating one or more fairness metrics into the model optimization functions in a bid to converge towards a model parameterization that maximizes performance and fairness.
	3. **Post-processing** approaches recognize that the actual output of an ML model may be unfair to one or more protected variables and/or subgroup(s) within the protected variable. Thus, post-processing approaches tend to apply transformations to model output to improve prediction fairness. Post-processing is one of the most flexible approaches, as it only needs access to the predictions and sensitive attribute information without requiring access to the actual algorithms and ML models. This makes them applicable for black-box scenarios where the entire ML pipeline is not exposed.

Advantage of pre-processing and post-processing methods is that they do not modify the ML method explicitly. This means that (open source) ML libraries can be leverages unchanged for model training. But **no direct control over the optimization function of the ML model itself.** 

**Modification of data/model may lead to less interpretability.**  [192, 202] from Fairness survey 

4. **Measuring Fairness and Bias**
	1. **Group Fairness Metrics**: compare the outcome of the classification algorithm for two or more groups. Commonly, these groups are defined through the sensitive variable.
		1. **Parity-based metrics**: typically consider predicted positive rates, i.e., $Py(\hat y = 1)$
			1. **Statistical/demograpic parity**: $Pr(\hat y = 1 | g_i) = Pr(\hat y=1 | g_j)$ that is, equal probability of being classified with the positive label. **Disadvantage:** the potential differences between groups are not being taken into account. 
			2. **Disparate Impact**: Considers the ratio between unprivileged and privileged groups. $\frac{Pr(\hat y = 1 | g_1)}{Pr(\hat y = 1 | g_2)}$ 
		2. **Confusion matrix based metrics**: Consider additional aspects like TPR, TNR, FPR, FNR. **Advantage**: they can include underlying differences between groups who would otherwise not be included in the parity-based approaches. 
			1. **Equal Opportunity**: promotes that the TPR is the same across different groups.
				1. $Pr(\hat y = 1 | y=1, g_i) = Pr(\hat y=1 | y=1, g_j)$ 
			2. **Equalized Odds**: in addition to TPR, simultaneously considers FPR as well, i.e., the percentage of actual negatives that are predicted as positive.
				1.  $Pr(\hat y = 1 | y=1, g_i) = Pr(\hat y=1 | y=1, g_j)$  &  $Pr(\hat y = 1 | y=0, g_i) = Pr(\hat y=1 | y=0, g_j)$  
			3. **Overall Accuracy Equality**: Look at the realtive accuracy rates across different groups. If two groups have the same accuracy, then tehy area considered equal based on their accuracy. 
				1. $\frac{TP_{g_i} + TN_{g_i}}{TP_{g_i} + TN_{g_i} + FP_{g_i} + FN_{g_i}}$  = $\frac{TP_{g_j} + TN_{g_j}}{TP_{g_j} + TN_{g_j} + FP_{g_j} + FN_{g_j}}$ 
			4. **Conditional User Accuracy Equality**: Do not look at the overall accuracy for each subgroup, but rather at the positive and negative predicted values. 
				1. $Pr(y = 1 | \hat y = 1, g_i) = Pr(y=1 | \hat y = 1, g_j)$  &  $Pr(y = 0 | \hat y = 0, g_i) = Pr(y=0 | \hat y = 0, g_j)$ 
			5. **Treatment Equality**: Considers the ratio of **False Negative Predictions (FNR)** to **False Positive Predictions (FPR)** 
				1. $\frac{Pr(\hat y = 1 |y=0, g_i)}{Pr(\hat y = 0 | y=1, g_i)} = \frac{Pr(\hat y = 1 |y=0, g_j)}{Pr(\hat y = 0 | y=1, g_j)}$  
			6. **Equalizing disincentives**: Compares two metrics: TPR and FPR
				1. $Pr(\hat y = 1 |y=1, g_i) - Pr(\hat y = 1 |y=0, g_i) = Pr(\hat y = 1 |y=1, g_j) - Pr(\hat y = 1 |y=0, g_j)$ 
			7. **Conditional Equal Opportunity**: As some metrics can be dependent on the underlying data distribution, Reference [30] provides an additional metric that specifies equal opportunity on a specific attribute a out of a list of attributes A, where τ is a threshold value
				1. $Pr(\hat y \geq \tau | g_i, y < \tau, A = a = Pr(\hat y \geq \tau | g_j, y < \tau, A = a$  
		3. **Calibration-based metrics**: calibration-based metrics take the predicted probability, or score, into account, differentiating them from metrics above that use predicted and actual values.
			1. **Test Fairness/calibration/matching conditional frequencies**: wants to guarantee that the probability of y = 1 is the same given a particular score; i.e., when two people from different groups get the same predicted score, they should have the same probability of belonging to y = 1.
				1. $Pr(y = 1 | S=s, g_i) = Pr(y=1 | S=s, g_j)$ 
			2. **Well Calibration**: extension of regular calibration where the probability of being in he positive class also has to be euqal to the particular score.
				1. $Pr(y = 1 | S=s, g_i) = Pr(y=1 | S=s, g_j) = s$ 
		4. **Score-based Metrics**
			1. **Balance for positive and negative class**: The expected predicted score for the positive and negative class has to be equal for all groups.
				1. $E(S=s | y=1, g_i) = E(S=s | y=1, g_j), E(S=s | y=0, g_i) = E(S=s | y=0, g_j)$ 
			2. **Bayesian Fairness** 
	2. **Individual and Counterfactual Fairness Metrics**: As compared to group-based metrics that compare scores across different groups, individual and counterfactual fairness metrics do not focus on comparing two or more groups as defined by a sensitive variable, but consider the outcome for each participating individual.
		1. **Counterfactual Fairness**: P(yˆA←a(U ) = y|X = x, A = a) = P(yˆA←a′ (U ) = y|X = x, A = a). Essentially, the definition ensures that changing an individual’s sensitive variable, while holding all other variables that are not causally dependent on the sensitive variable constant, does not change the prediction (distribution).
		2. **Generalized Entropy Index**: considers differences in an individual’s prediction ($b_i$) to the average prediction accuracy (μ). It can be adjusted based on the parameter $\alpha$ where $b_i = \hat y_i -y_i +1$ and $\mu = \frac{\Sigma_i b_i}{n}$ 
			1. $GEI = \frac{1}{n\alpha(\alpha -1 )}\Sigma_{i=1}^n [(\frac{b_i}{\mu})^\alpha - 1]$  
		3. **Theil Index**: GEI where alpha = 1
			1. $Theil = \frac{1}{n}\Sigma_{i=1}^n(\frac{b_i}{\mu})log(\frac{b_i}{\mu})$   
5. **Binary classification approaches**
	1. *Building on the metrics discussed above, fairness in ML researchers seek to mitigate unfairness by “protecting” sensitive variables. The literature is dominated by approaches for mitigating bias and unfairness in ML **within the problem class of binary classification** [26]. There are many reasons for this, but most notably: (1) many of the most contentious application areas that motivated the domain are binary decisions (hiring vs. not hiring; offering a loan vs. not offering a loan, etc.); (2) quantifying fairness on a binary dependent variable is mathematically more convenient; addressing multi-class problems would add terms to the fairness quantity.* 
	2. There is no strategy which does all in-pre-and post-processing interventions together. 
	3. **Blinding**: Fairness objective: Make a classifier “immune” to one or more sensitive variables
		1. . Whereas omission refers to not including the sensitive variables as input for the prediction models, immunity also considers the indirect effect that sensitive variables can have on other (input) variables of a prediction model. For instance, sensitive variables often are correlated with other variables in the data, and approaches focusing on immunity aim to prevent these indirect effects from resulting in discrimination measured through the sensitive variable.
		2. Disadvantages
			1. Omission has been shown to decrease model accuracy and increase discrimination
			2. Both omission and immunity overlook relationships with proxy variables
	4. **Causal Methods**: Fairness Objective: Identify potentially useful relationships between sensitive and non-sensitive variables to provide insights for fairness-related methodological decisions.
		1. A key objective is to uncover causal relationships in the data and find dependencies between sensitive and non-sensitive variables
		2. Disadvantage
			1. Requires significant computational resources
	5. **Sampling and Subgroup Analysis**: **Fairness Objective**: Sampling methods have two primary objectives: (1) to create samples for the training of robust algorithms (e.g., References [9, 43, 56, 90, 148, 263, 283, 307]), i.e., “correct” training data and eliminate biases [148]; and (2) to identify groups (or subsamples) of the data that are significantly disadvantaged by a classifier, i.e., as a means to evaluate a model (e.g., References [2, 70, 315]).
		1. Challenge: sufficient data should be available for each subgroup. 
	6. **Transformation**: Fairness Objective: Learn or generate new “fair” representations of the data (e.g., a mapping or projection function) that still preserves the fidelity of the ML task
		1. Challenges
			1. The transformed data should not be significantly different from the original data, otherwise the extent of “repair” can diminish the utility of the produced classifier [88, 98, 202] through data loss [125]
			2. Understanding the relationship(s) between sensitive and potential proxy variables is hard [98], thus causal methods (Section 4.2) may be useful precursor to transformation techniques.
			3. The selection of “fair” target distributions is not straightforward
			4. Finding an “optimal” transformation under high dimensionality can be computationally expensive, even under assumptions of convexity
			5. Missing data provides specific problems for transformation approaches, as it is unclear how to deal with such data samples. Many handle this by simply removing these samples, yet this may raise other methodological issues.
			6. Transformation makes the model less interpretable [192, 202], which may be at odds with data protection legislation.
			7. There are no guarantees that the transformed data have “repaired” discriminatory latent relationships with proxy variables
	7. **Relabelling and Perturbation**: Fairness Objective: Modify the training data such that underprivileged and privileged instances are treated similarly and/or explore the effects of such modifications on model fairness.
	8. **Reweighing**: Fairness Objective: Change the “impact” of instances (observations) on the prediction model during training to promote “fair(er)” handling of sensitive variables and/or underprivileged groups.
		1. Unlike transformation, relabelling, and perturbation approaches that alter (certain instances of) the data, reweighing assigns weights to instances of the training data while leaving the data themselves unchanged.
	9. **Regularization and Constraint Optimisation**: Extend the classifier’s loss function such that it penalizes “unfair” outcomes.
	10. **Adversarial Learning**: Fairness Objective: Tutor a classification model to be “fairer” by providing in-training feedback or modifying the training data to promote immunity to one or more sensitive variables.
	11. **Calibration**: Fairness Objective: To adjust the probability outputs of a model such that the portion of predicted positive outcomes matches that of positive examples across (or within) all (sub)groups in the dataset.
	12. **Thresholding**: Fairness Objective: Consider fairness metrics in the setting of thresholds for predicted scores (or decision boundaries in general) produced by an ML model.
