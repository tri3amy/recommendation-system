# 🛍️ E-Commerce Product Recommendation System
**Content-Based Filtering using TF-IDF and K-Nearest Neighbors (KNN)**

![Status](https://img.shields.io/badge/Status-Completed-success)
![App](https://img.shields.io/badge/App-Streamlit-red)
Python, Scikit-Learn, Pandas, NLTK, SciPy, Streamlit.

## Repository Structure

```text
recommendation-system/
│
├── data/
│   ├── Reviews.csv           # Raw dataset (Ignored from GitHub due to >100MB size)
│   └── products.csv          # Generated clean data (Ignored)
│
├── models/
│   └── neighbors.npz         # Saved KNN distances & indices (Ignored)
│
├── app.py                    # Streamlit web application (Frontend & Inference)
├── preprocess.py             # Data cleaning, NLP processing, & Model Training
├── environment.yml           # Conda environment dependencies
├── requirements.txt          # Pip environment dependencies
├── .gitignore                # Ignoring large data and model files
└── README.md                 # Project documentation
```

## Project Overview
* **Problem Statement:** In an e-commerce platform, users often struggle to find products similar to the ones they already like.
* **Objective:** This project aims to build a Content-Based Recommendation System that suggests the top 5 similar products based on textual reviews, average product ratings, and review helpfulness scores.
* **Tech Stack:** Python, Pandas, Scikit-Learn (TF-IDF, KNN, MinMaxScaler), NLTK (SnowballStemmer, Stopwords), and Streamlit for the interactive web interface.

## How It Works (The Pipeline)

### 1. Data Preprocessing & Feature Engineering (`preprocess.py`)
Since the raw dataset contains over 500,000 reviews, the data is first filtered and processed:
* **Filtering:** Only products with a minimum of 10 reviews are kept to ensure robust recommendations.
* **Feature Engineering:** Calculated `avg_score` and `helpfulness_ratio` for each product.
* **NLP Processing:** Combined 'Summary' and 'Text' reviews into a single corpus per product. Applied lowercasing, regex cleaning, NLTK English stopwords removal, and Snowball Stemming.
* **Vectorization:** Converted the cleaned text into numerical features using **TF-IDF Vectorizer** (max 1000 features).
* **Combining Features:** Merged the sparse TF-IDF matrix with MinMax scaled numerical features (`avg_score` and `helpfulness_ratio`) using `scipy.sparse.hstack`.
* **Model Training:** Trained a **Nearest Neighbors** model using the **Cosine Similarity** metric. The calculated distances and indices are saved as a compressed `.npz` file for fast inference.

### 2. Interactive Web App (`app.py`)
A user-friendly web interface built with **Streamlit**.
Instead of recalculating TF-IDF and training the model upon every launch, the app efficiently loads the pre-computed `products.csv` and `neighbors.npz`. When a user selects a product from the dropdown, the app instantly retrieves and displays the top 5 nearest neighbors (recommendations) along with their best review summaries and average ratings.

---

## How to Run the Project Locally
**⚠️ IMPORTANT NOTE:** The raw dataset (`Reviews.csv`) and the generated model files are **not included** in this repository because their sizes exceed GitHub's 100MB limit. You must download the dataset and run the preprocessing script first before launching the app.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/tri3amy/recommendation-system.git](https://github.com/tri3amy/recommendation-system.git)
cd recommendation-system
```

### Step 2: Set up Virtual Environment & Install Dependencies
You can use either Conda or standard Python `venv` to set up the environment.

**Option A: Using Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate rsystem_env
```

**Option B: Using standard Python (pip)**
```bash
python -m venv env

# On Windows:
env\Scripts\activate

# On Mac/Linux:
source env/bin/activate

pip install -r requirements.txt
```

### Step 3: Download the Dataset
1. Download the dataset from Kaggle: [Amazon Product Reviews Dataset](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset).
2. Extract the downloaded archive.
3. Place the `Reviews.csv` file inside the `data/` folder in this project directory.

### Step 4: Run the Preprocessing Script
Run this script to clean the data, perform TF-IDF vectorization, train the KNN model, and generate the necessary files.

```bash
python preprocess.py
```
*Wait for the process to finish. It will generate `products.csv` inside the `data/` folder and `neighbors.npz` inside the `models/` folder.*

### Step 5: Launch the Streamlit App
Once the preprocessing is complete, you can start the interactive web application:

```bash
streamlit run app.py
```
*The app will open automatically in your browser at `http://localhost:8501`.*

---

Author: Tri Puji Utami  
**LinkedIn:** 
