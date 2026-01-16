# Language & AI Project — Author Profiling Pipeline (Thinking & Extrovert)

This repository contains the end-to-end workflow used for our assignment: **sentiment reduction → EDA/cleaning → modeling + evaluation**, applied to the **Thinking** and **Extrovert** datasets provided by the course.

## Project Overview

We followed a staged pipeline to ensure consistent preprocessing and to keep training feasible on limited memory:

1. **Download datasets** (Thinking and Extrovert) from the professor’s provided link.
2. **Run Sentiment Reduction** on both datasets and save the processed outputs.
3. **Run EDA + Cleaning** using the sentiment-reduced files and save the cleaned datasets.
4. **Run Modeling notebooks**, where training is split across separate notebooks per text representation and category.

We chose to train **per file/notebook** for efficiency and memory reasons.
Each directory has the code dedicated for each dataset, namely EXTROVERT_CODE has the extrovert dataset code and THINKING_CODE the thinking dataset code. Note that the code for both datasets is the same to ensure fairness when comparing.
---

## Repository Structure (High-level)

- `sentiment_reduction/`  
  Code/notebooks to apply sentiment reduction and save processed datasets.

- `Data_cleaning_splitting/`  
  Code/notebooks for exploratory data analysis, cleaning and splitting our datasets for modeling. Inputs are the sentiment-reduced files, outputs are cleaned datasets.

- `modeling/`  
  Modeling notebooks. Training is organized by **label/category** (Thinking if its for the thinking dataset) and **text representation**:
  - **Raw text**
  - **Stopword-removed text**
  - **Lemmatized text**
  That means that THINKING_author_profiling_model_lemmatization is the model of the lemmatized thinking dataset

Each representation has its own modeling notebook to improve **runtime** and reduce **RAM/usage**, since each notebook trains one configuration at a time.


## How to Run the Pipeline

### Step 0 — Download the datasets
Download the **Thinking** and **Extrovert** datasets from the courses’s link.

Place them in the expected input folder (adapt to your repo paths)

### Step 1 — Sentiment Reduction
Run the sentiment reduction code/notebook on both datasets. This produces sentiment-reduced files.

### Step 2 — EDA and Cleaning
Run the EDA/cleaning code using the sentiment-reduced outputs as input. This produces cleaned datasets and the splits

### Step 3 — Modeling (separate notebooks per representation)
Run the modeling notebooks. Each dataset/category has separate notebooks for:
- raw text
- stopword version
- lemmatized version



---

## Notes on Outputs

Model evaluation plots and metrics are generated within the modeling/evaluation notebooks

---

## Contribution Summary (Team)

- **Aleksandra**: initiated the evaluation code implementation  
- **Yağmur**: implemented the sentiment reduction component  
- **Alicia**: led the core modeling code
- **Maria**: data cleaning/EDA and support for model/evaluation code 

---

## Requirements

See `requirements.txt` (or the notebook setup cells) for the required Python packages.

---

## Reproducibility

To reproduce results successfully:
- Run the pipeline **in order** (sentiment reduction → cleaning → modeling).
- Ensure the saved intermediate files are placed in the expected folders.
- Keep notebook-specific settings (e.g., text column name, label column name, model choice) consistent with the folder naming.


