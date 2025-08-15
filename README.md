# 🛍️ H&M Personalized Fashion Recommendation System

## 📌 Overview

This project builds a personalized recommendation system to predict which fashion items users aged **20–40** are most likely to buy within the **next 7 days**, using transactional and product metadata from the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) dataset on Kaggle.

We plan to use **collaborative filtering** for the baseline model and then implement an **LSTM (Long Short-Term Memory)** network to capture sequential purchase patterns. The goal is to explore how well machine learning models can provide relevant, diverse, and timely product recommendations in a real-world retail setting.

---

## 🧠 Research Question

**How accurately can machine learning models predict which fashion items users aged 20–40 will purchase within a 7-day period, using transaction history and product metadata from the H&M dataset?**

---

## 📊 Dataset

**Source:** [Kaggle - H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

Key files:
- `transactions.csv`: User purchase history with timestamps
- `articles.csv`: Product metadata (e.g., category, color, garment type)
- `customers.csv`: Customer metadata (age, fashion preferences, etc.)

> We filter customers to those aged **20–40** for this study.

---

## 📚 Methods & Models

1. **Exploratory Data Analysis (EDA)**  
   Understand user behavior, popular items, and purchase frequency.

2. **Collaborative Filtering (Baseline)**  
   - Matrix factorization
   - Item-based and user-based similarity models (e.g. `Surprise`, `LightFM`)

3. **LSTM Recommender (Advanced)**  
   - Sequence modeling of user transactions
   - Learns purchase patterns over time
   - Predicts next likely item(s) in the sequence

4. **Evaluation Metrics**
   - Precision@K, Recall@K
   - Hit Rate, MAP
   - Diversity & Novelty (optional)

