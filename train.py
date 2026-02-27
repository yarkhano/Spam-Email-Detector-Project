import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

#Libraries for proper performance surance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

# DATA LOADING & CLEANING , Data Preprocessing
dataset = pd.read_csv('spam_mails_dataset.csv')

# Drop unnecessary columns
new_dataset = dataset.drop(columns=["Unnamed: 0", "label_num"])

# Rename columns for clarity
new_dataset.rename(columns={"label": "category", "text": "message"}, inplace=True)

# Map text labels to numbers (0 for ham, 1 for spam)
new_dataset["category"] = new_dataset["category"].map({"ham": 0, "spam": 1})

#removing the irrelevant words symbols , that do not affect predictions of model
new_dataset["message"] = new_dataset["message"].str.replace(r"^Subject:\s*","", regex=True, case=False)
new_dataset["message"] = new_dataset["message"].str.replace(r"^(re|fw)\s*:\s*", "", regex=True, case=False)
new_dataset["message"] = new_dataset["message"].str.replace(r"[^\w\s]", " ", regex=True)
new_dataset["message"] = new_dataset["message"].str.lower()   #Force everything to lowercase, so our tfdfvectorizer understand e.g Free,free & Free same word and do not create multiple columns
new_dataset["message"] = new_dataset["message"].str.replace(r"\s+", " ", regex=True).str.strip()

print("Data Head after cleaning:")
print(new_dataset.head(5))

#  VECTORIZATION (The Translator)
# We define our 'X' (features) and 'y' (target) here so the split function can find them
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(new_dataset["message"])
y = new_dataset["category"]

# --- 3. DATA SPLITTING ---
# Now that X and y exist, we can split them into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. MODEL TRAINING ---
# We use SVM (Support Vector Machine) which is excellent for text classification
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# --- 5. EVALUATION ---
# We must generate predictions (y_pred) to check the accuracy and create the confusion matrix
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results in a clean, professional format
print("\n--- MODEL PERFORMANCE METRICS ---")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}% (No False Alarms)")
print(f"Recall:    {recall * 100:.2f}% (Catching all the Spam)")
print(f"F1-Score:  {f1 * 100:.2f}% (Overall Balance)")

# --- 6. VISUALIZATION (Confusion Matrix) ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Spam Detection Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# --- 7. SAVING THE ASSETS ---
# Save the 'Brain' (SVM Model)
model_filename = 'spam_mail_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"\nModel saved as {model_filename}")

# Save the 'Translator' (TF-IDF Vectorizer)
# IMPORTANT: You cannot use the model later without this specific translator
vectorizer_filename = 'spam_vectorizer.pkl'
with open(vectorizer_filename, 'wb') as file:
    pickle.dump(tfidf, file)
print(f"Vectorizer saved as {vectorizer_filename}")
print("Training Pipeline Complete! 🎉")