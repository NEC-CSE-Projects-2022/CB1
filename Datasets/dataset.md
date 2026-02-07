# Dataset Documentation

## Dataset Title
CICIDS-2017

## Usage of Dataset
In this work, the CICIDS-2017 dataset is used as the primary benchmark for developing and evaluating the proposed FuzzTabIDS system. It contains about 2.8 million network flow records with 84 features and multiple attack categories.

The dataset is first cleaned, preprocessed, and normalized by removing identifiers, handling missing values, and encoding labels. After preprocessing, around 70 numerical features are retained, and MRMR (Minimum Redundancy Maximum Relevance) is applied to select the top 30 most relevant features.

The dataset is used for three classification tasks: binary (benign vs attack), 7-class grouped, and 15-class multi-class detection. A stratified 75:25 train-test split, along with cross-validation and temporal validation, is used to ensure reliable evaluation.

The processed data is then passed through TabNet, fuzzy logic, correction models, ensemble learning, and XGBoost stages to train and test the intrusion detection pipeline.

## Dataset Information
The CICIDS-2017 dataset is a publicly available intrusion detection benchmark developed by the Canadian Institute for Cybersecurity.

It contains about 2.8 million labeled network flow records collected over five days (July 3–7, 2017) in a simulated enterprise network environment.

Each record includes 84 features such as flow identifiers, time-based metrics, statistical attributes, and protocol-related information.

The dataset covers both normal and malicious traffic, with around 2.27 million benign samples and 5.56 lakh attack samples belonging to 15 different attack types, including DoS, DDoS, PortScan, Bot, Web attacks, Infiltration, and Heartbleed.

It supports binary, grouped, and multi-class intrusion detection tasks.

## Dataset Name
CICIDS-2017

## Source
https://www.unb.ca/cic/datasets/ids-2017.html  
https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

## Domain
Network Intrusion Detection  
Cyber Security  
Intrusion Detection System

## Task
The task is network intrusion detection through supervised classification.

The system learns from labeled network traffic data to automatically classify each network flow as either normal or malicious and identify the type of attack.

Classification tasks:

1. Binary Classification  
   Benign vs Attack

2. 7-Class Classification  
   Grouped attack categories

3. 15-Class Classification  
   Exact attack identification

## Problem Type
Supervised machine learning classification.

Includes:
- Binary classification
- Multi-class classification
- Class imbalance
- High-dimensional data

## File Format
CSV (Comma Separated Values)

## Dataset Link
https://www.unb.ca/cic/datasets/ids-2017.html  
https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

## Dataset Overview
The CICIDS-2017 dataset is a benchmark dataset for intrusion detection research.

It contains about 2.8 million records with 84 features and 15 attack types collected in a realistic environment.

It is highly imbalanced and supports both binary and multi-class classification.

## Project Summary
This study uses the CICIDS-2017 dataset to build and evaluate the FuzzTabIDS system.

After preprocessing, around 70 features are retained and MRMR selects the top 30 features.

The data passes through TabNet, fuzzy logic, correction models, ensemble learning, and XGBoost stages.

The final system achieves high detection accuracy and improved minority-class performance.
