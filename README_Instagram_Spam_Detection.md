# 🧠 Instagram Spam Detection

A Machine Learning project that detects **spam/fake Instagram accounts** using user profile data and behavioral features.  
This project demonstrates end-to-end training, testing, and evaluation of classification models like **Random Forest** and **Logistic Regression**.

---

## 📁 Project Structure

```
📦 Instagram-Spam-Detection
├── train.csv                # Training dataset
├── test.csv                 # Testing dataset
├── instagram_spam_detection.py  # Training script
├── instagram_spam_pipeline.pkl  # Saved trained model
└── README.md                # Project documentation
```

---

## 🚀 Features

- Preprocesses numeric Instagram profile data  
- Trains and evaluates ML models  
- Compares **RandomForestClassifier** and **LogisticRegression**  
- Exports the best-performing model for prediction  

---

## ⚙️ Installation

Make sure you have Python 3.8+ installed, then install the dependencies:

```bash
pip install pandas scikit-learn joblib
```

---

## 🧩 Model Training

Run the training script to build and save the model:

```bash
python instagram_spam_detection.py
```

This script:
- Loads `train.csv`
- Trains both Random Forest and Logistic Regression models
- Selects the best model (based on F1-score)
- Saves it as `instagram_spam_pipeline.pkl`

---

## 📊 Model Evaluation

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|-----------|
| **Random Forest** | ~0.93 | ~0.91 | ~0.92 | ~0.91 |
| **Logistic Regression** | ~0.89 | ~0.87 | ~0.88 | ~0.87 |

*(Values are examples — your results depend on dataset.)*

🟢 **Selected Model:** Random Forest Classifier  
📈 **Reason:** Higher F1-score, better balance of precision and recall.

---

## 🔍 Usage for Prediction

Once the model is trained, you can load it to classify new accounts:

```python
import joblib, pandas as pd

# Load trained model
pipe = joblib.load("instagram_spam_pipeline.pkl")

# Load new Instagram data
new_data = pd.read_csv("new_data.csv")

# Predict spam/genuine
predictions = pipe.predict(new_data)
print(predictions)
```

---

## 🧠 Tech Stack

- **Language:** Python 🐍  
- **Libraries:** Scikit-learn, Pandas, Joblib  
- **Algorithm:** RandomForestClassifier, LogisticRegression  

---

## 🗂️ Dataset Info

Both `train.csv` and `test.csv` contain profile-level data such as:
- Followers count  
- Following count  
- Number of posts  
- Engagement ratio  
- Bio completeness  
- Verified status  
- Target column: **`fake`** (1 = Spam/Fake, 0 = Genuine)

---

## 🏁 Output

After running the script, you’ll get:
- `instagram_spam_pipeline.pkl` → trained ML model  
- Classification report & confusion matrix in console  
- Predictions for new unseen data  

---

## 👩‍💻 Author

**Tanvi Kashyap**  
📍 ABVGIET Pragatinagar, Shimla  
🧑‍🏫 Submitted to **Ms. Krishika Thakur**  
🏢 Under the company **Isekai Tech**

---

## 📜 License

This project is open-source and free to use for academic or research purposes.

---

⭐ *If you like this project, give it a star on GitHub!*  
