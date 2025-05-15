# Multiclass-Multilabel-prediction-For-stack-overflow-Questions
This is my project that I have made for Edvancer Eduventures , for the assignment submission. This is first project in the course Deep Learning Expert (Part of IITK AI) 

# StackOverflow Tag Prediction System

This Python script implements a machine learning pipeline to predict tags for StackOverflow questions based on their text content (title, body, and answers).

## Purpose

The system aims to:
- Automatically suggest relevant tags for new StackOverflow questions
- Demonstrate a multi-label text classification approach
- Handle large datasets efficiently with memory optimization techniques

## Code Overview

### Key Features
- **Data Processing**:
  - Combines questions, answers, and tags from StackOverflow datasets
  - Cleans text by removing HTML tags and special characters
  - Efficiently samples large datasets (45% in this implementation)

- **Model Pipeline**:
  - Uses TF-IDF for text vectorization (limited to 3000 features)
  - Implements One-vs-Rest logistic regression for multi-label classification
  - Handles class imbalance with balanced class weights

- **Memory Optimization**:
  - Implements progressive data loading
  - Includes memory tracking utilities
  - Explicit garbage collection

### Output Files

1. **`merged_dataset_45percent.csv`**  
   Contains the processed dataset combining questions, answers, and tags (45% sample)

2. **`test_predictions.csv`**  
   Contains test data with both predicted and actual tags for evaluation

3. **`stackoverflow_tag_predictor_45percent.pkl`**  
   Saved model artifacts including:
   - Trained pipeline (TF-IDF + Logistic Regression)
   - MultiLabelBinarizer for tag encoding
   - Top tags used in modeling
   - Classification report from testing

## Performance Metrics

The system evaluates model performance using:
- Classification report (precision, recall, f1-score per tag)
- Hamming loss (fraction of incorrect labels)

## Usage Notes

- Requires the following input files in `DATA_DIR`:
  - `Questions.csv`
  - `Answers.csv`
  - `Tags.csv`

- Configuration options at the top of the script allow adjusting:
  - Sample fraction
  - Minimum samples per tag
  - Number of tags to predict
  - Model hyperparameters
 
## Dataset Sampling and Memory Management
Due to hardware limitations and the large size of the complete StackOverflow dataset, this implementation uses only 45% of the available data (controlled by the SAMPLE_FRACTION = 0.45 parameter) to prevent system crashes during processing. The memory-efficient sampling is implemented in two key sections:

```python
def load_and_sample(filename, usecols):
    # First pass to get row count
    with open(DATA_DIR / filename, 'r', encoding='ISO-8859-1') as f:
        n_rows = sum(1 for _ in f) - 1
    
    # Calculate rows to skip using random sampling
    skip = sorted(np.random.choice(
        np.arange(1, n_rows+1),
        size=int(n_rows * (1 - SAMPLE_FRACTION)),
        replace=False
    ))
    
    # Read CSV while skipping selected rows
    return pd.read_csv(...)

```

This function first counts total rows, then randomly selects a subset to load, significantly reducing memory usage.



Memory Monitoring

```python
def print_memory_usage():
    print(f"Memory used: {psutil.virtual_memory().percent}%")

# Explicit garbage collection calls throughout:
gc.collect()
```

The script includes periodic memory checks and manual garbage collection to manage resources. Additional optimizations include:

Deleting unused columns immediately after processing

Using float32 instead of float64 in the TF-IDF vectorizer

Limiting feature space to 3000 dimensions

While this sampling approach reduces the training data volume, it maintains representative tag distributions through stratified sampling where possible (Lines 198-206).



## Dependencies

```python
pandas, numpy, scikit-learn, tqdm, psutil, re, pickle, pathlib

```


## model_training_code

```python

# Import necessary libraries
# pandas for data manipulation
# re for regular expressions
# pickle for saving model artifacts
# numpy for numerical operations
# sklearn modules for machine learning pipeline
# tqdm for progress bars
# psutil for memory tracking
# gc for garbage collection
# warnings to suppress warnings
import pandas as pd
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, hamming_loss
from tqdm import tqdm
import psutil
import gc
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Initialize tqdm for pandas to show progress bars for pandas operations
tqdm.pandas()

# Function to print current memory usage
def print_memory_usage():
    print(f"Memory used: {psutil.virtual_memory().percent}%")

# Function to clean text data by removing HTML tags, special characters, and normalizing whitespace
def clean_text(text):
    try:
        if pd.isna(text) or text == '':
            return ""
        text = str(text)
        text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s.,;?!]', ' ', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.lower().strip() or "emptytext"  # Convert to lowercase
    except Exception as e:
        print(f"Error cleaning text: {str(e)}")
        return "error_text"

# Print initialization message and current memory usage
print("Initializing...")
print_memory_usage()

# Configuration settings
# DATA_DIR: Path to the directory containing input files
# SAMPLE_FRACTION: Fraction of data to use (45% in this case)
# MIN_SAMPLES_PER_TAG: Minimum number of samples required to keep a tag
# TOP_N_TAGS: Number of top tags to keep for modeling
DATA_DIR = Path("C:/Users/Mudassir/Desktop/Edvancer Assignment Submission/Python 2/Assignment 1/")
SAMPLE_FRACTION = 0.45  # Use 45% of data
MIN_SAMPLES_PER_TAG = 10  # Increased minimum samples for better quality
TOP_N_TAGS = 10  # Number of tags to keep

# Function to load and sample data efficiently by skipping random rows
def load_and_sample(filename, usecols):
    try:
        # First pass to get row count
        with open(DATA_DIR / filename, 'r', encoding='ISO-8859-1') as f:
            n_rows = sum(1 for _ in f) - 1  # Subtract header
        
        # Calculate rows to skip using random sampling
        skip = sorted(np.random.choice(
            np.arange(1, n_rows+1),
            size=int(n_rows * (1 - SAMPLE_FRACTION)),
            replace=False
        ))  # Fixed missing parenthesis here
        
        # Read CSV while skipping selected rows
        return pd.read_csv(
            DATA_DIR / filename,
            encoding='ISO-8859-1',
            usecols=usecols,
            skiprows=skip
        )
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return pd.DataFrame()

# Load and sample the three input files: Questions, Answers, and Tags
print("\nLoading and sampling questions...")
questions = load_and_sample('Questions.csv', ['Id', 'Title', 'Body'])
print("\nLoading and sampling answers...")
answers = load_and_sample('Answers.csv', ['ParentId', 'Body'])
print("\nLoading and sampling tags...")
tags = load_and_sample('Tags.csv', ['Id', 'Tag'])

# Validate that all data files were loaded successfully
if questions.empty or answers.empty or tags.empty:
    raise ValueError("Failed to load one or more data files")

# Clean text data in questions and answers
print("\nProcessing text data...")
questions['cleaned_text'] = (questions['Title'].astype(str) + ' ' + 
                           questions['Body'].progress_apply(clean_text))
answers['cleaned_body'] = answers['Body'].progress_apply(clean_text)

# Free memory by deleting unused columns and running garbage collection
del questions['Body'], answers['Body']
gc.collect()

# Select top tags based on frequency and minimum sample threshold
print("\nSelecting top tags with sufficient samples...")
tag_counts = tags['Tag'].value_counts()
top_tags = tag_counts[tag_counts >= MIN_SAMPLES_PER_TAG].index.tolist()[:TOP_N_TAGS]
print(f"Selected tags: {top_tags}")

# Filter tags to only include the selected top tags
tags_filtered = tags[tags['Tag'].isin(top_tags)]
question_tags = tags_filtered.groupby('Id')['Tag'].apply(list).reset_index()

# Merge questions with their tags
print("\nMerging datasets...")
merged_data = pd.merge(questions, question_tags, on='Id', how='inner')

# Process answers by combining all answers for each question
answers_grouped = answers.groupby('ParentId')['cleaned_body'].agg(' '.join).reset_index()
answers_grouped.columns = ['Id', 'combined_answers']
final_data = pd.merge(merged_data, answers_grouped, on='Id', how='left')

# Create final text feature by combining question text and answers
print("\nCreating final text feature...")
final_data['full_text'] = final_data['cleaned_text'] + ' ' + final_data['combined_answers'].fillna('')
final_data = final_data[final_data['full_text'].str.len() > 0]

# Save merged dataset to CSV
merged_data_path = "merged_dataset_45percent.csv"
final_data.to_csv(merged_data_path, index=False)
print(f"\nMerged dataset saved to {merged_data_path}")

# Free memory by deleting unused DataFrames
del questions, answers, merged_data, answers_grouped
gc.collect()

# Prepare for modeling by binarizing multi-label tags
print("\nPreparing for modeling...")
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(final_data['Tag'])
X = final_data['full_text']

# Print class distribution
print("\nClass distribution:")
class_dist = pd.Series(y.sum(axis=0), index=mlb.classes_)
print(class_dist)

# Ensure we have at least 2 samples per class for stratification
valid_tags = y.sum(axis=0) >= 2
y = y[:, valid_tags]
mlb.classes_ = np.array(mlb.classes_)[valid_tags]

# Split data into training and test sets with stratification if possible
print("\nTrain-test split...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    print("Using random split due to small class sizes")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# Create a lightweight pipeline with TF-IDF vectorizer and logistic regression
print("\nSetting up lightweight pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=3000,  # Reduced features for memory efficiency
        ngram_range=(1,1),  # Only unigrams
        dtype='float32',    # Use float32 to save memory
        min_df=3,          # Lower min_df to keep more terms
        max_df=0.85       # Ignore very common terms
    )),
    ('clf', OneVsRestClassifier(
        LogisticRegression(
            solver='liblinear',  # Good for small datasets
            max_iter=200,       # Reduced iterations for speed
            class_weight='balanced',  # Handle class imbalance
            random_state=42     # For reproducibility
        )
    ))
])

# Train the model
print("\nTraining model...")
pipeline.fit(X_train, y_train)

# Evaluate the model on test data
print("\nEvaluating model...")
y_pred = pipeline.predict(X_test)

# Print classification report and hamming loss
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
print("Hamming Loss:", hamming_loss(y_test, y_pred))

# Create a DataFrame with test data, predictions, and actual values
print("\nCreating output with predictions and actual values...")
# Get the original test data rows
test_data = final_data.loc[X_test.index].copy()

# Convert predicted labels back to tag names
predicted_tags = mlb.inverse_transform(y_pred)
test_data['predicted_tags'] = [' '.join(tags) for tags in predicted_tags]

# Convert actual labels back to tag names
actual_tags = mlb.inverse_transform(y_test)
test_data['actual_tags'] = [' '.join(tags) for tags in actual_tags]

# Save the test data with predictions to a CSV file
output_filename = "test_predictions.csv"
test_data.to_csv(output_filename, index=False)
print(f"\nSaved test data with predictions to {output_filename}")

# Save model artifacts including the pipeline and label binarizer
print("\nSaving model artifacts...")
model_artifacts = {
    'model': pipeline,
    'mlb': mlb,
    'top_tags': top_tags,
    'test_report': classification_report(y_test, y_pred, 
                                      target_names=mlb.classes_,
                                      output_dict=True),
    'sample_fraction': SAMPLE_FRACTION
}

with open("stackoverflow_tag_predictor_45percent.pkl", 'wb') as f:
    pickle.dump(model_artifacts, f)

print("\nDone! Model saved as 'stackoverflow_tag_predictor_45percent.pkl'")
print("Final memory usage:")
print_memory_usage()


```


## Code Explanation 

####  Importing Necessary Libraries

```python
import pandas as pd
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, hamming_loss
from tqdm import tqdm
import psutil
import gc
import warnings
from pathlib import Path

```
Libraries used for data processing, machine learning, memory monitoring, and more.


####  Initialize Environment


```python
tqdm.pandas()
warnings.filterwarnings('ignore')

```
Enables progress bars on pandas operations.
Suppresses unnecessary warnings.


####  Utility Functions


```python
def clean_text(text):
    try:
        if pd.isna(text) or text == '': return ""
        text = re.sub(r'<[^>]+>', ' ', str(text))
        text = re.sub(r'[^\w\s.,;?!]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip() or "emptytext"
    except Exception as e:
        print(f"Error cleaning text: {str(e)}")
        return "error_text"

```
Prints the memory usage 
Cleans HTML tags, special characters, and normalizes whitespace.



####  Configuration Settings

```python
DATA_DIR = Path("C:/Users/Mudassir/.../Assignment 1/")
SAMPLE_FRACTION = 0.45
MIN_SAMPLES_PER_TAG = 10
TOP_N_TAGS = 10
```

Controls input location, sampling, and tag filtering.




####   Load and Sample Data

```python
def load_and_sample(filename, usecols):
    ...
```
Efficiently loads a random subset of rows.
Handles encoding and file read errors.


####   Load Questions, Answers, Tags

```python
questions = load_and_sample('Questions.csv', ['Id', 'Title', 'Body'])
answers = load_and_sample('Answers.csv', ['ParentId', 'Body'])
tags = load_and_sample('Tags.csv', ['Id', 'Tag'])
    ...
```
Loads the three core datasets.



####   Text Cleaning and Merging

```python
questions['cleaned_text'] = questions['Title'] + questions['Body'].progress_apply(clean_text)
answers['cleaned_body'] = answers['Body'].progress_apply(clean_text)
    ...
```
Applies clean_text() to both questions and answers.



####   Tag Filtering and Merging

```python
tag_counts = tags['Tag'].value_counts()
top_tags = tag_counts[tag_counts >= MIN_SAMPLES_PER_TAG].index.tolist()[:TOP_N_TAGS]
...
final_data = pd.merge(...)
    ...
```
Selects top N tags.
Merges questions and answers.



####   Create Final Text Feature

```python
tag_counts = tags['Tag'].value_counts()
top_tags = tag_counts[tag_counts >= MIN_SAMPLES_PER_TAG].index.tolist()[:TOP_N_TAGS]
...
final_data = pd.merge(...)
    ...
```

Combines question and answer text into one field for modeling.




####   Model Preparation

```python
y = MultiLabelBinarizer().fit_transform(final_data['Tag'])
X = final_data['full_text']
```
Transforms tag labels into binary vectors.




####   Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(...)
```
Splits data with stratification if possible.



####   Model Training and Evaluation

```python
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(...))
```
Trains and evaluates the model.


####   Save Predictions

```python
test_data['predicted_tags'] = ...
test_data.to_csv("test_predictions.csv")
```
Converts binary predictions back to tag names.
Saves output with predictions.


####   Save Model Artifacts

```python
model_artifacts = {
    'model': pipeline,
    'mlb': mlb,
    'top_tags': top_tags,
    ...
}
with open("stackoverflow_tag_predictor_45percent.pkl", 'wb') as f:
    pickle.dump(model_artifacts, f)
```
Stores the model, label binarizer, and configuration.



####   Final Memory Check

```python
print_memory_usage()
```



##  Model Performance Metrics
The model saved as stackoverflow_tag_predictor_45percent.pkl was evaluated using standard classification metrics. The table below summarizes its performance across the top 10 most frequent StackOverflow tags:

| Tag         | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| android     | 0.78      | 0.93   | 0.85     | 3,618   |
| c#          | 0.66      | 0.91   | 0.77     | 4,072   |
| c++         | 0.57      | 0.92   | 0.70     | 1,849   |
| html        | 0.34      | 0.86   | 0.48     | 2,467   |
| ios         | 0.77      | 0.92   | 0.84     | 1,815   |
| java        | 0.66      | 0.89   | 0.76     | 4,739   |
| javascript  | 0.55      | 0.88   | 0.68     | 5,045   |
| jquery      | 0.47      | 0.90   | 0.62     | 3,219   |
| php         | 0.75      | 0.91   | 0.82     | 4,066   |
| python      | 0.83      | 0.94   | 0.88     | 2,584   |

Observations:
The classifier demonstrates strong performance on tags like python, android, and ios with F1-scores above 0.84.

Tags such as html, c++, and javascript show lower precision, indicating some label noise or semantic overlap with other tags.

Overall, the model achieves high recall across all tags, ensuring minimal missed predictions, which is important for multi-label classification.

The trained model is saved as stackoverflow_tag_predictor_45percent.pkl

## Dropbox Link for output file 

This is the dropbox link for the output prediction csv files :

test_predictions ,
merged_dataset_45percent

https://www.dropbox.com/home/Multiclass-Multilabel-prediction-For-stack-overflow-Questions
