import pandas as pd
import numpy as np

dataset=pd.read_csv('spam_mails_dataset.csv')

dataset.shape

new_dataset=dataset.drop(columns=["Unnamed: 0","label_num"])

new_dataset.rename(columns={"label":"category","text":"message"}, inplace=True)
new_dataset["category"] = new_dataset["category"].map({"ham":0,"spam":1})


new_dataset.head()

from sklearn.feature_extraction.text import CountVectorizer
vc=CountVectorizer()

y=new_dataset["category"]
X=vc.fit_transform(new_dataset["message"]).toarray()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr=LogisticRegression()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

# 1. Import the  Vectorizer (tfidf)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 2. Setup TF-IDF (It automatically penalizes common words)
# We still remove 'english' stop words to be safe
tfidf = TfidfVectorizer(stop_words='english')

# 3. Create X and y
X = tfidf.fit_transform(new_dataset["message"]).toarray()
y = new_dataset["category"]

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)


def predict_email(text):

    x = tfidf.transform([text]).toarray()
    prediction = model.predict(x)
    return "🚨 SPAM" if prediction[0] == 1 else "✅ Safe (Ham)"

print("Lunch Check:", predict_email("it is to inform you that tomorwo is off."))
print("Spam Check:", predict_email("Congratulations! You won a free iPhone."))


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True)
plt.show()


# import joblib
# joblib.dump(model,'spam_model.pkl')
# joblib.dump(tfidf,'vectorizer.pkl')