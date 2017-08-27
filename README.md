<hr>
# Machine Learning Concepts
	## Approaches
		- Learning classifier systems
		- Rule-based machine learning
		- Genetic algorithms
		- Sparse dictionary learning
		- Similariy and metric learning
		- Representation learning
		- Reinforcement learning
		- Bayesian networks
		- Clustering
		- Support vector machines
		- Inductive logic programming
		- Deep learning
		- Artificial neural networks
		- Decision tree learning
		- Association rule learning
	## Motivation
		- Prediction
		- Inference
	## Performance Analysis
		- Goodness of Fit = R^2
		- Bias-Variance Tradeoff
		- ROC Curve - Receiver Operating Characteritics
		- Confusion Matrix
		- Accuracy
		- f1 Score
			- Recall
			- Precision
			- Harmonic Mean of Precision and Recall
		- Mean Squared Error (MSE)
		- Error Rate
	## Taxonomy
	## Selection Criteria
	## Types
		- Regression
		- Classifiaction
		- Clustering
		- Density Estimation
		- Dimensionality Reduction
	## Tuning
		- Bagging
		- Underfitting
		- Early Stopping
		- Cross-validation
		- Hyperparameters
		- Overfitting
		- Bootstrap
	## Libraries
		- Python
			- Microsoft Cognitive Toolkit
			- Torch
			- Keras
			- MXNet
			- Scikit-learn
			- Numpy
			- Pandas
			- Tensorflow
	## Categories
		- Supervised
		- Unsupervised
		- Reinforcement Learning
	## Kind
		- Parametric
		- Non-Parametric
<hr>
# Machine Learning Process
	## Data
		- Build Datasets
			- Machine Learning is math. In specific, performing Linear Algebra on Matrices. Our data values must be numeric.
		- Encode Features
		- Select Features
		- Engineer Features
		- Impute Features
		- Find
		- Collect
		- Explore
		- Clean Features
	## Question
		- Reinforcement Learning: What should I do now?
		- Anomaly Detection: Is this anomalous?
		- Classification: Is this A or B?
		- Regression: How much, or how many of these?
		- Clustering: How can these elements be grouped?
	## Model
		- Select Algorithm based on question and data available
	## Cost Function
		- The cost function will provide a measure of how far my algorithm and its parameters are from accurately representing my training data.
		- Sometimes referred to as Cost or Loss function when the goal is to minimise it, or Objective function when the goal is to maximise it.
	## Results and Benchmarking
	## Optimization
		- Having selected a cost function, we need a method to minimise the Cost function, or maximise the Objective function when the goal is to maximise it.
		- The cost function will provide a measure of how far my algorithm and its parameters are from accurately representing my training data.
	## Direction
		- Machine Learning Research
		- SaaS: Pre-built Machine Learning models
			- ...many others
			- AWS
				- Polly
				- Rekognition
				- Lex
			- Google Cloud
				- Language API
				- Jobs API
				- Vision API
				- Speech API
				- Video Intelligence API
				- Translation API
		- Data Science and Applied Machine Learning
			- Tools: Jupiter/Datalab/Zeppelin
			- Google Cloud
				- ML Engine
			- AWS
				- Amazon Machine Learning
			- ...many others
	## Tuning
		- Different Algorithms have different Hyperparameters, which will affect the algorithms performance. There are multiple methods for Hyperparameter Tuning, such as Grid and Random search.
	## Scaling
		- How does my algorithm scale for both training and inference?
	## Deployment and Operationalisation
		- How can feature manipulation be done for training and inference in real-time?
		- How to make sure that the algorithm is retrained periodically and deployed
	## Infrastructure
		- Is the infrastructure adapter to be the algorithm we are running? Should GPUs be confidered rather than CPUs?
		- Can the infrastructure running the machine learning process scale?
		- How is access to the ML algorithm provided? REST API? SDK?

<hr>
# Machine Learning Models
	## Instance Based
		- Self-Organising Map (SOM)
		- k-nearest Neighbour (kNN)
		- Learning Vecor Quantization (LVQ)
		- Locally Weighted Learning (LWL)
	## Decision Tree
		- Random Forest
		- Classification and Regression Tree (CART)
		- Gradient Boosting Machines (GBM)
		- Condistional Decision Trees
		- Gradient Boosted Regression Trees (GBRT)
	## Clustering
		- Validation
			- Data Structure Metrics
				- Silhouette Width
				- Dunn Index
				- Connectivity
			- Stability Metrics
				- Average Distance Between Means ADM
				- Non-overlap APN
				- Average Distance AO
				- Figure of Merit FOM
		- Algorithms
			- DBScan
			- Expectation Maximization
			- k-Medians
			- Hierarchical Clustering
				- Dissimilarity Measure
					- Euclidean
					- Manhattan
				- Linkage
					- Average
					- Complete
					- Single
					- Centroid
			- k-Means
				- How many clusters do we select?
			- Fuzzy C-Means
			- Self-Organising Maps (SOM)
	## Nueral Networks
		- Unit (Neurons)
		- Inut Layer
		- Hidden Layers
		- Batch Normalization
		- Learning Rate
		- Weight Initialization
			- All Zero Initialization
			- Linitialization with Smal Random Numbers
			- Calibrating the Variances
		- Backpropogation
		- Activation Functions
			- Defines the output of that node giveen ian input or set of inputs
			- Types
				- Softplus
				- Binary
				- ReLU
				- Sigmoid/Logistic
				- Tanh
				- Softmax
				- Maxout
	## Regression
		- Logistic Regression
			- Logistic Function
		- Least Absolute Shrinkage and Selection Operator (LASSO)
		- Ridge Regression
		- Locally Estimated Scatterplot Smoothing (LOESS)
		- Linear Regression
		- Generalised Linear Models (GLMs)
	## Dimensionality Reduction
		- Linear Discriminant Analysis (LDA)
		- Quadratic Discriminant Analysis (QDA)
		- Partial Least Squares Regression (PLSR)
		- Principal Component Analysis (PCA)
		- Partial Least Squares Regression (PLSR)
		- Partial Least Squares Discriminant Analysis
	## Bayesian
		- Bayesian Belief Network (BBN)
		- Naive Bayes
			- Naive Bayes Classifier
		- Multinominal Naive Bayes

<hr>
# Machine Learning Mathematics
	## Cost/Lost(Min) Objective(Max) Functions
		- Kullback-geibier Divergence
		- Exponential
		- 0-1 Loss
		- Logistic
		- Maximum Likelihood Estimation (MLE)
		- Cross-Entropy
		- Quadratic
		- Hinge Loss
		- Hellinger Distance
		- Itakura-Saito distance
	## Information Theory
		- Mutual Information
		- Joint Entropy
		- Entropy
		- Cross Entropy
		- Conditional Entropy
		- Kullback-Leibler Divergence
	## Statistics
		- Central Limit Theorem
		- Techniques
		- Relationship
		- Measures of Central Tendency
		- Dispersion
	## Probability
		- Concepts
			- Chain Rule
			- Law of Total Probability
			- Bayes Theorem (rule, law)
			- Random Variable
			- Frequentist vs Bayesian Probability
				- Frequentist
				- Bayesian
			- Independence
			- Conditionality
			- Marginalization
	## Linear Algebra
		- Curse of Dimensionality
		- Gradient
		- Derivatives Chain Rule
		- Matrices
		- Elgenvectors and Elgenvalues
		- Jacobin Matric
		- Tensors
	## Optimization
		- Mini-batch Stochastic Gradient Descent
		- Momentum
		- Gradient Descent
		- Stochastic Gradient Descent
		- Adagrad
	## Density Estimation
		- Methods
			- Cubic Spline
			- Kernel Density Estimation
				- Uniform, Triangle, Quartic, Triweight, Gaussian, Cosine, others...
				- non-parametric
				- real-valued
				-  non-negative
				- it's a type of PDF that it is symmetric
				- integral over functionis equal to 1
				- calculates kernel distributions for every sample point, and then adds all the distributions
		- Postly Non-parametric. Parametric makes assumpsions on my data/ranbome variables fro instance that hey are normally distributed. Non-parametric doesnot.
		- The methods are generally intended for description rather than formal inference.
	## Regularization
		- Mean=constrained regularization
		- Sparse regularizer on columns
		- Early Stopping
		- L1 norm
		- L2 norm
		- Dropout
		- Nuclear norm regularization
		- Clustered mean=constrained regularization
		- Graph-based similarity
	## Distributions
		- Cumulative Distribution Function (CDF)
		- Definition
		- Type (Density Function)

<hr>
# Machine Learning Data Processing
	## Feature Engineering
		- Reframe Numerical Quantities
		- Decompose
		- Discretization
			- Continuous Features
			- Categorical Features
		- Crossing
	## Dataset Construction
		- Validation Dataset
			- A set of parameters used to tune the parameters of a classifier
		- Training Dataset
			- A set of examples used for learning.
		- Test Dataset
			- A set of examples used only to assess the performance of a fully trained classifier
		- Cross Validation
	## Data Exploration
		- Univarriate Analysis
			- Continuous Features
			- Categorical Features
		- Bi-variate Analysis
			- .
				- ANOVA
				- Z-Test/T-Test
				- Chi-Square Test
				- Stacked Column Chart
				- Two-way table
			- Correlation Plot - Heatmap
			- Scatter Plot
			- Finds out the relationship between two variables
		- Variable Identification
			- Identify Predictor (input) and target (output) vaiables. Next, identify the data type and category of the variables
	## Feature Selection
		- Importance
			- Embedded Methods
				- Lasso regression
				- Ridge regression
			- Filter Methods
				- ANOVA: Analysis of Variance
				- Correlation
				- Linear Discriminant Analysis
				- Chi-Square
			- Wrapper Methods
				- Recursive Feature Elimination
				- Forward Selection
				- Backward Elimination
				- Genetic Algorithms
		- Dimensionality Reduction
			- Principle Component Analysis
			- Singular Value Decomposition (SVD)
		- Correlation
			- Features should be uncorrelated with each other and highly correlated to the feature we're trying to predict.
	## Data Types
		- Ratio - has all the properties of an interval variable, and also has a clear definition of 0.0.
		- Interval - is a measurement where the difference between two values is meaningful.
		- Ordinal - is one where the order matters but not the difference between values.
		- Nominal - is for mutual exclusive, but not ordered, categories
	## Feature Encoding
		- Label Encoding
			- One Hot Encoding
				- In One Hot Encoding, make sure the encodings are done in a way that all features are linearly independent.
		- Machine Learning algorithms perform Linear Algebra on Matrices, which means all features must be numeric. Encoding helps us do this.
	## Feature Cleaning
		- Outliers
		- Missing values
		- Special values
		- Obvious inconsistencies
	## Feature Normalsation or scaling
		- Since the rancge of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. Another resaon why feature scaling is applied is that gradient descent converges much faster with feature scaling whan without it.
		- Methods
			- Rescaling
			- Standardization
			- Scaling to unit length
	## Feature Imputation
		- Some Libraries...
		- Mean-substitution
		- Hot-deck
		- Cold-deck
		- Regression