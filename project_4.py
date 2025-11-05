# Step 1: Import libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 2: Create a simple dataset
emails = [
    "Win a $1000 Walmart gift card now!",
    "Your meeting is scheduled for 10 AM tomorrow.",
    "Congratulations, you have won free tickets!",
    "Please find the attached project report.",
    "Earn money quickly from home!!!",
    "Can we reschedule our appointment?",
    "Limited offer! Buy 1 get 1 free!",
    "Your invoice for last month is attached."
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]  
# 1 = spam, 0 = not spam

# Step 3: Convert text to numeric features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Step 5: Train a model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Test model accuracy
y_pred = model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Try predicting new emails
new_emails = [
    "Win cash prizes instantly!",
    "Can you send me the report by today?"
]
new_features = vectorizer.transform(new_emails)
predictions = model.predict(new_features)

for email, label in zip(new_emails, predictions):
    print(f"\nEmail: {email}")
    print("Prediction:", "SPAM" if label == 1 else "NOT SPAM")
