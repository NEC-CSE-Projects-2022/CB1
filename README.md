
# Team Number – Project Title

## Team Info
- 22471A05I4 — **VEERA DHANUSH KUMAR PEKALA** ( [LinkedIn](www.linkedin.com/in/dhanush-pekala-a531b52b1) )
_Work Done: xxxxxxxxxx_

- 22471A05I6 — **SRINIVAS SEGGEM** ( [LinkedIn](https://www.linkedin.com/in/srinivas-seggem-78aa44362 ) )
_Work Done: xxxxxxxxxx_

- 22471A05J1 — **SAIDA VALI MYNAMPATI SK** ( [LinkedIn](https://www.linkedin.com/in/m-saidavali-shaik-6b5b912ab) )
_Work Done: xxxxxxxxxx_

- 22471A05K2 — **VENKATA SIVA THANNARU** ( [LinkedIn](https://www.linkedin.com/in/shiva2213) )
_Work Done: xxxxxxxxxx_

---

## Abstract
Most of the current Intrusion Detection Systems (IDS) fail to be accurate under high-dimensional features, class imbalance, and imprecise predictions—especially under subtle or emerging attack patterns. This contribution presents FuzzTabIDS, a twelve-stage hybrid IDS pipeline that combines feature selection, interpretable deep learning, fuzzy logic, andensemble corrections to achieve robust and adaptive intrusion detection. Starting with the CICIDS 2017 dataset, we use MinimumRedundancy Maximum Relevance (MRMR) to choose the mostrelevant features. Next, a TabNet classifier is trained to generateclass labels as well as confidence scores. For managing low confidence predictions, a fuzzy logic layer dynamically determinesdecision thresholds and activates micro-correction modules, suchas Decision Trees and Random Forests, for fine grained tuning.Ensemble of TabNet, LightGBM, and Random Forest throughweighted voting and stacking further improves detection accuracy. A final-stage XGBoost corrector reduces remaining errors inedge cases. Evaluation of experiments for binary, multi-class, andall class configurations shows high accuracy, enhanced recall onminority classes, and robust interpretability, proving FuzzTabIDSto be a good solution for contemporary network security issues.

---

## Paper Reference (Inspiration)
👉 **[Paper Title "Streamlined Network Intrusion Detection: Feature Selection
Optimization for Higher Accuracy and Efficiency"
  – Author Names Kunda Suresh Babu, Vallepu Srinivas, Yamani Chandana, G Satish, R Dhandu Naik and
Dodda Venkata Reddy
 ](https://www.taylorfrancis.com/chapters/edit/10.1201/9781003515470-16/streamlined-network-intrusion-detection-feature-selection-optimization-higher-accuracy-efficiency-kunda-suresh-babu-vallepu-srinivas-yamani-chandana-satish-dhandu-naik-dodda-venkata-reddy)**
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
Improved accuracy, precision, recall over multi and all class labelling. 
improved overall accuracy on all three types of classifications.
reduced the features required for training and get high accuracy.

---

## About the Project
What your project does
Think of FuzzTabIDS as a smarter intrusion detection system. It watches network traffic and decides whether something is normal or an attack. But instead of depending on a single model, it uses a layered strategy: feature selection, an interpretable deep-learning model, fuzzy logic to handle uncertainty, and multiple correction/ensemble steps to nail down the final prediction. The goal is simple: catch attacks with extremely high accuracy, especially the tricky ones that most IDS models usually miss.

Why it is useful
Here’s the thing: real network traffic is messy. You get millions of records, lots of redundant features, and many rare attacks that mimic normal behavior. Most IDS models struggle when data becomes high-dimensional or when predictions fall into “confusion zones” (like when the model outputs 0.49 vs 0.51 probability).

our system solves these pain points by:
Cutting out noisy features using MRMR so the model focuses on what actually matters.
Using TabNet, which doesn’t just classify but also explains which features influenced the decision.
Adding fuzzy logic so the system doesn’t blindly trust borderline predictions.
Using correction layers and an ensemble so even uncertain or minority-class attacks get rescued.
Ending with an XGBoost “final judge” that fixes any last errors.
The result is a far more reliable and high-recall IDS, with near-perfect performance reported across binary, 7-class, and full 15-class tasks. 


General project workflow

1. Input
You start with the CICIDS-2017 dataset: ~2.8M network flow records filled with more than 80 statistical and behavioral features. These include IP/port info, packet sizes, flow durations, etc.

2. Processing
Before anything touches a model, you:
Clean and normalize the data
Remove useless or constant columns
Encode all labels (for binary, 7-class, and 15-class settings)
Select the top 30 features using MRMR to reduce noise and redundancy
This creates a leaner, cleaner dataset for the models to learn from.

3. Model Pipeline
This is where things get interesting. The system runs in multiple stages:
TabNet gives the first prediction + confidence score.
Fuzzy logic checks: Is this prediction uncertain?
If yes, send it to Decision Tree and Random Forest micro-correction modules.
Combine all predictions using a weighted ensemble and then a stacked ensemble.
Any remaining hard cases go to a final XGBoost corrector, which patches the last errors.
Each stage improves the prediction—exactly as shown in the ablation table on page 6 of the document, where accuracy climbs step by step. 

4. Output
The system returns:
Final predicted class (benign or type of attack)
Near-perfect accuracy, recall, and F1 across all configurations
Interpretable feature attributions thanks to TabNet’s attention masks



---

## Dataset Used
👉 **[CICIDS 2017 DATASET](https://www.unb.ca/cic/datasets/ids-2017.html)**

**Dataset Details:**
Name:
CICIDS-2017

What it contains:
Network traffic records collected over five days, with both normal activity and many types of cyber-attacks.

Total samples:
About 2.8 million network flow entries.

Number of features:
84 columns per sample (after cleaning, your project uses 70).
These features describe things like packet sizes, flow duration, connection behavior, and protocol stats.

Types of data included:
Benign traffic (normal user behavior)
15 different attacks, including DoS, DDoS, PortScan, Brute Force, Botnet, Web attacks, Infiltration, Heartbleed, SQL Injection, etc.

Class distribution:
Benign: 2,271,320 samples
Attacks: 556,556 samples


---

## Dependencies Used
Data Handling & Preprocessing:
pandas – for loading and cleaning the CICIDS-2017 dataset
numpy – numeric operations, handling missing/inf values
scikit-learn –
LabelEncoder
Train-test split
MinMaxScaler
DecisionTreeClassifier
RandomForestClassifier
LogisticRegression
MRMR feature selection (via mutual information)

Deep Learning:
TabNet (pytorch-tabnet) – main classifier used in the first stage

Gradient Boosting Models:
LightGBM – part of the weighted ensemble
XGBoost – final residual error corrector
Evaluation & Metrics
scikit-learn metrics – accuracy, precision, recall, F1, ROC curves
matplotlib / seaborn – for confusion matrices and plots

Other utilities:
Imbalanced data handling tools (noise injection for RF booster)
Learning rate schedulers (from PyTorch for TabNet)

---

## EDA & Preprocessing
1. EDA (Exploratory Data Analysis) –
Checked dataset size
~2.8 million records and 84 columns (features + labels).
Looked at class distribution
Benign: 2,271,320
All attacks combined: 556,556
Some attacks are very frequent (DoS Hulk: 230,124), some are extremely rare (SQL Injection: 21, Heartbleed: 11).
This clearly shows heavy class imbalance.

Understood feature types
Mix of:
Flow identifiers (IP, ports, flow IDs)
Time features (duration, inter-arrival times)
Statistical features (mean, std, min/max of bytes/packets)
Behavioral/protocol flags

High dimensional + redundant features
With 80+ features, many are redundant, constant, or not useful, which motivated using MRMR feature selection.


2. Preprocessing – Step by step
Here’s the workflow your paper describes, in plain language:

a) Column cleanup
You removed columns that can cause data leakage or don’t help learning, such as:

Flow ID, Timestamp

Source/Destination IP

Source/Destination Port

Protocol, SourceFile

These fields are too specific and don’t generalize well for an IDS model. 



b) Handle useless / bad features
Dropped constant columns (same value for all rows).

Replaced any inf / -inf with NaN.

Imputed missing numeric values using the median.

Dropped rows with >20% missing values to keep data quality high.

c) Label cleaning & encoding
You prepared labels for three tasks:

Binary

BENIGN → 0

Any attack → 1

7-class grouped (to reduce imbalance)

BENIGN → 0

All DoS types → 1

PortScan → 2

DDoS → 3

Brute Force (FTP/SSH) → 4

Bot → 5

Web attacks + Infiltration + Heartbleed → 6

15-class (all attacks separately)
Each unique attack type gets its own label ID (0–14).

Encoding done with LabelEncoder / manual mapping depending on the setup. 


d) Normalization
All numerical features were scaled using Min–Max Scaling so that they fall roughly in the same range (e.g., 0 to 1). This helps TabNet and other models train faster and not be dominated by large-scale features.


---

## Model Training Info
1. Training the Main Model: TabNet

Training setup
Batch size: 1024

Ghost batch size: 128

Optimizer: Adam

Learning rate: 0.02

LR scheduler: StepLR for gradual decay

Early stopping: patience of 15 epochs

TabNet also produces confidence scores, which are later used by the fuzzy logic layer.

2. Fuzzy Logic Layer Training

Analyze TabNet’s confidence distribution

Detect a fuzzy zone around the 0.5 probability

Dynamically set thresholds

Example:

T_low = 0.5 − Δ

T_high = 0.5 + Δ

Δ varies (0.03 to 0.1) depending on imbalance and validation F1 trends.

This layer basically learns how to treat uncertain predictions.

3. Micro-Correction Models
These are small models trained only on fuzzy-zone errors (samples where TabNet was unsure, usually 0.3–0.7 probability).

a) Decision Tree Corrector
Shallow tree (max_depth = 4)

Trained on borderline misclassified samples

Purpose: correct simple mistakes cheaply

b) Random Forest Booster
Trained on uncertain + misclassified samples

Extra Gaussian noise added to increase training data

Helps recover minority-class errors (rare attacks)

4. Ensemble Training
All three base models—TabNet, LightGBM, and Random Forest—are trained separately.

a) Weighted Voting
Weights:

TabNet → 0.5

LightGBM → 0.3

Random Forest → 0.2

Weighted prediction = sum of (model probability × weight)

b) Stacked Ensemble
A Logistic Regression model is trained on:

TabNet probabilities

LightGBM probabilities

Random Forest probabilities

This meta-model learns how to combine predictions more intelligently.

5. Final Step: XGBoost Residual Corrector
XGBoost trains only on the remaining misclassified cases after the ensemble.

Its job:

Handle hardest edge-cases

Improve recall on rare classes

Clean up final errors before output

This is the step that pushes accuracy and F1 very close to 99.99% across tasks 


---

## Model Testing / Evaluation
Every sample in the test set goes through the full pipeline in this order:

TabNet predicts the class and confidence.

Fuzzy logic checks whether the confidence is inside the “uncertain zone”.

If uncertain → send to

Decision Tree corrector

Random Forest booster

Ensemble models (TabNet + LGBM + RF) combine their predictions:

Weighted voting

Then stacked Logistic Regression

XGBoost final corrector fixes any remaining misclassifications.

The final output is the class prediction after all corrections.

---

## Results
1. Binary Classification Results
Final performance (after XGBoost correction):
Accuracy: 99.99%

Precision: 1.00

Recall: 1.00

F1-score: 1.00

The confusion matrix (page with Fig. 1) shows almost no misclassifications in the final stage. 


2. 7-Class Classification Results
This setup groups related attacks into 7 major categories to address imbalance.

Final performance (after XGBoost correction):
Accuracy: 99.99%

Precision: 1.00

Recall: 1.00

F1-score: 1.00

The 7-class confusion matrix (Fig. 3) shows perfect classification across all attack groups. 


3. 15-Class (All Attack Types) Results
This is the most challenging configuration because many attack types have extremely few samples (e.g., SQL Injection = 21, Heartbleed = 11).

Even here, the system performs extremely well.

Final performance (after XGBoost correction):
Accuracy: 99.99%

Precision: 0.99

Recall: 0.99

F1-score: 0.99

---

## Limitations & Future Work
Despite its outstanding performance, FuzzTabIDS still has some practical challenges. The entire pipeline appears more complex than a one-model IDS, that might require considerations of optimization for run-time deployment. For example, although, the XGBoost booster layers greatly improve
precision and recall, they might add latencies in streaming scenarios. Furthermore, the current experiments concentrate
only on the CICIDS-2017 data set. While that’s a good
benchmark, it is an open area for exploration whether the
system can generalize to unseen datasets like BoT- IoT or real
world enterprise logs. In the future our plans involve extensive
evaluation of system with other benchmark datasets such as
NSL-KDD and BoT-IoT, evaluating system on streaming and
real time settings and optimizations of the pipeline for mini
mumlatency deployment. Efforts will also be made to improve
explainability and incorporate online learning mechanisms. We
conclude that FuzzTabIDS offers a promising candidate as a
resilient, interpretable, uncertainty-aware intrusion detection
system for a dynamic network security space.

---

## Deployment Info
REST API (Flask)
Load all models into memory
Receive JSON with flow features
Return predicted class + confidence


---
