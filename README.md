# Phishing Email Detection Bot (Enron Dataset)
**Author:** Mohammad Ali Othman

## Overview
This project builds a machine learning pipeline to detect **phishing-like emails** using the **Enron Emails Dataset**.  
It combines:
- **Feature Engineering** (email metadata + text patterns + URL indicators)
- **TF-IDF** for subject/body text representation
- **Dimensionality Reduction (Truncated SVD)** for high-dimensional TF-IDF features
- **Clustering (KMeans / MiniBatchKMeans)** to group similar email patterns
- A **Neural Network classifier (PyTorch)** to predict phishing probability

Because the Enron dataset is not labeled for phishing, a **rule-based auto-labeling** strategy is used to create a realistic “possible phishing” target for training.

---

## Key Features
- Extensive phishing-oriented feature engineering:
  - Urgency, threat, prize/money phrases
  - Credential request indicators
  - Sender spoofing patterns
  - URL-based risk features (shorteners, many subdomains, @ symbol, etc.)
  - Text statistics (length, punctuation, uppercase ratio, overlap, etc.)
  - Cyclical time encoding from the email date
- Weighted TF-IDF for email subject
- TruncatedSVD for TF-IDF dimensionality reduction
- KMeans clustering used as an additional risk-pattern signal
- Neural Network model trained with:
  - Undersampling to handle class imbalance
  - Early stopping and learning-rate scheduling
- Saved model artifacts:
  - `best_phishing_model.pth`
  - `phishing_detection_pipeline.pkl`

---

## Project Structure


├── notebooks/

│ ├── training_pipeline.ipynb

│ ├── graph_representation.ipynb

│ └── Bot_for_detecting_phishing_email.ipynb

├── src/

│ ├── features.py

│ ├── model.py

│ ├── pipeline.py

│ ├── training_pipeline.py

│ ├── email_client.py

│ └── graph.py

└── README.md



---

## Methodology
### 1) Data Preparation
- Load Enron emails from CSV
- Clean missing values and remove duplicates
- Parse email timestamps and apply cyclical encoding (sin/cos)

### 2) Feature Engineering
A large set of features is extracted from:
- Subject, body, sender fields
- CC/BCC counts and recipient count patterns
- Keyword-based phishing indicators (urgency, threats, impersonation)
- URL patterns and risk signals

### 3) Text Representation
- TF-IDF on **subject** (weighted)
- TF-IDF on **body**
- Concatenate engineered features + TF-IDF

### 4) Unsupervised Pattern Discovery
- Apply Truncated SVD to reduce TF-IDF dimensionality
- Use KMeans / MiniBatchKMeans clustering
- Use cluster membership as an additional behavioral signal

### 5) Supervised Classification
- Build a PyTorch neural network to predict phishing probability
- Train with undersampling to reduce majority-class dominance
- Evaluate using accuracy, precision, recall, F1-score, and AUC-ROC

---

## Outputs
- **Saved pipeline** (`phishing_detection_pipeline.pkl`): vectorizers, scaler, SVD, clustering model, NN weights
- **Model weights** (`best_phishing_model.pth`)
- **Plots / evaluation figures** (e.g., confusion matrix, precision-recall curve)
- Optional **phishing network graph visualization** (senders → suspicious domains)

---

## Notes
- The labeling approach is **heuristic** and designed to approximate phishing behavior in real life.
- Results depend on the dataset distribution and the chosen threshold for auto-labeling.
- This project can be extended by using a real labeled phishing dataset and adding header-level features.

---

## License
This project is for educational and research purposes.
