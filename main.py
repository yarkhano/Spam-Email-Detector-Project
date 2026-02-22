import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# DATA LOADING & CLEANING
dataset = pd.read_csv('spam_mails_dataset.csv')

# Drop unnecessary columns
new_dataset = dataset.drop(columns=["Unnamed: 0", "label_num"])

# Rename columns for clarity
new_dataset.rename(columns={"label": "category", "text": "message"}, inplace=True)

# Map text labels to numbers (0 for ham, 1 for spam)
new_dataset["category"] = new_dataset["category"].map({"ham": 0, "spam": 1})

# --- 2. VECTORIZATION (The Translator) ---
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
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- 6. VISUALIZATION (Confusion Matrix) ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Spam Detection Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# --- 7. PREDICTION FUNCTION ---
def predict_email(text):
    # We use the 'tfidf' object we trained earlier to transform new input
    x_input = tfidf.transform([text])
    prediction = model.predict(x_input)
    return "🚨 SPAM" if prediction[0] == 1 else "✅ Safe (Ham)"

# Quick Tests
print("Lunch Check:", predict_email("it is to inform you that tomorwo is off."))
print("Spam Check:", predict_email("Congratulations! You won a free iPhone. Click here to claim."))

# --- 8. SAVING THE ASSETS ---
# Save the 'Brain' (SVM Model)
model_filename = 'spam_mail_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved as {model_filename}")

# Save the 'Translator' (TF-IDF Vectorizer)
# IMPORTANT: You cannot use the model later without this specific translator

vectorizer_filename = 'spam_vectorizer.pkl'
with open(vectorizer_filename, 'wb') as file:
    pickle.dump(tfidf, file)
print(f"Vectorizer saved as {vectorizer_filename}")


#Load the model
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)
    print(f"model loaded as {model_filename}")

with open(vectorizer_filename, 'rb') as file:
    loaded_vectorizer = pickle.load(file)
    print(f"vectorizer loaded as {vectorizer_filename}")


#verifying the loaded model is working correctly.
sample_message = "Information regarding submission of form"
transformed_message = loaded_vectorizer.transform([sample_message])
predicted_result = loaded_model.predict(transformed_message)
if predicted_result == 1:
    print("This email is spam.")
elif predicted_result == 0:
    print("This email is safe.")
print(f"message: {sample_message}")
print(f"prediction: {predicted_result}")

