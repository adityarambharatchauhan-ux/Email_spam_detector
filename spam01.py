{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fc69f007-bc78-4d44-8e71-af043889ce3a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data shape: (5000, 2)\n",
      "Columns: Index(['text', 'label'], dtype='object')\n",
      "Model Accuracy: 1.0\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "         ham       1.00      1.00      1.00       498\n",
      "        spam       1.00      1.00      1.00       502\n",
      "\n",
      "    accuracy                           1.00      1000\n",
      "   macro avg       1.00      1.00      1.00      1000\n",
      "weighted avg       1.00      1.00      1.00      1000\n",
      "\n",
      "\n",
      "Model and Vectorizer saved successfully!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "import pickle\n",
    "\n",
    "# 1. Load data - FIX for 1 column CSV\n",
    "df = pd.read_csv(\"SPAM.csv\", on_bad_lines='skip')\n",
    "\n",
    "# If CSV has 1 column, split it\n",
    "if df.shape[1] == 1:\n",
    "    df = df['text,label'].str.split(',', n=1, expand=True)\n",
    "    df.columns = ['text', 'label']\n",
    "\n",
    "print(\"Data shape:\", df.shape)\n",
    "print(\"Columns:\", df.columns)\n",
    "\n",
    "# 2. Separate features and labels\n",
    "x = df[\"text\"]\n",
    "y = df[\"label\"]\n",
    "\n",
    "# 3. Convert text to numbers using TF-IDF\n",
    "vectorizer = TfidfVectorizer(stop_words=\"english\", lowercase=True, max_features=5000)\n",
    "X = vectorizer.fit_transform(x)\n",
    "\n",
    "# 4. Split into train and test\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# 5. Train Naive Bayes model\n",
    "model = MultinomialNB()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# 6. Check accuracy\n",
    "y_pred = model.predict(X_test)\n",
    "print(\"Model Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n",
    "\n",
    "# 7. Save model and vectorizer\n",
    "pickle.dump(model, open('spam_model.pkl', 'wb'))\n",
    "pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))\n",
    "print(\"\\nModel and Vectorizer saved successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec635cea-556c-452b-a7c8-3a269ee68658",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
