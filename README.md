# CS-516_Project
# 🎯 Fairness in Targeted Advertisements  

> **“When algorithms speak, let them speak fairly.”**  
> This project builds an end-to-end pipeline to **detect**, **inject** (synthetic bias), and **mitigate** bias in ad-targeting systems—so your models can learn and serve, not discriminate.

---

## 📖 Table of Contents
1. [Why This Matters](#why-this-matters)  
2. [What You’ll Find Here](#what-youll-find-here)  
3. [How It Works](#how-it-works)  
4. [Getting Started](#getting-started)  
5. [Core Features](#core-features)  
6. [Results & Insights](#results--insights)  
7. [Roadmap & Next Steps](#roadmap--next-steps)  
8. [Contributing](#contributing)  
9. [License](#license)  
10. [Acknowledgments & Contact](#acknowledgments--contact)  

---

## 📌 Why This Matters
In today’s data-driven world, ad platforms can unintentionally reinforce stereotypes.  
By combining **real-world** (Meta Ad Library, FairJobs) and **synthetic** data, we shine a light on hidden biases—then apply and measure fairness interventions.  
Whether you’re a researcher, engineer, or ethics advocate, this repo helps you:
- Understand bias metrics  
- Experiment safely with controlled injections  
- Prototype and evaluate debiasing strategies  

---

## 🔍 What You’ll Find Here
- **Data Sources**  
  - Meta Ad Library exports  
  - FairJobs public dataset  
  - Configurable synthetic-data generator  
- **Scripts & Modules**  
  - `preprocess.py` for cleaning & feature engineering  
  - `data_gen_tunable.py` to craft biased or fair toy sets  
  - `train.py` + `evaluate.py` for model training & fairness testing  
- **Notebooks**  
  - Visual explorations of bias metrics vs. accuracy  
  - Step-by-step demo of mitigation techniques  
- **Results**  
  - Charts & tables showing trade-offs between fairness and performance  
  - Summaries of best-in-class mitigation approaches  

---

## ⚙️ How It Works
1. **Ingest & Clean**  
   - Parse raw CSV/API exports  
   - Standardize demographic & ad features  
2. **Inject Controlled Bias**  
   - Tune parameters in `config/bias_params.json`  
   - Generate synthetic cohorts with known disparity  
3. **Train & Mitigate**  
   - Baseline model (e.g., logistic regression)  
   - Apply reweighing, adversarial debiasing, disparate impact constraints  
4. **Evaluate**  
   - Compute Statistical Parity Difference, Disparate Impact Ratio, Accuracy  
   - Visualize with Jupyter notebooks  

---

## 🚀 Getting Started
```bash
# 1. Clone this repo
git clone https://github.com/yourusername/CS-516_Project.git
cd CS-516_Project

# 2. Set up a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run preprocessing
python src/preprocess.py --input data/fairjobs/raw.csv --output data/fairjobs/clean.csv

# 5. Generate synthetic data
python src/data_gen_tunable.py --params config/bias_params.json

# 6. Train & evaluate
python src/train.py --data-dir data/
python src/evaluate.py --results-dir results/


📈 Results & Insights
🚦 Statistical Parity
Reduced disparity by 93.4% on FairJobs with reweighing

🛡️ Adversarial Debiasing
Achieved a Disparate Impact Ratio > 0.99 without sacrificing >1% accuracy

🌍 Geo-Bias Correction
Demographic constraints trimmed location skew with only a 1% accuracy dip


Authors: Shruthi Kodati, Anand Meena, Parikha Goyanka
Mentors: Prof.  Abolfazl Asudeh(University of Illinois Chicago)

Thank you for visiting!

