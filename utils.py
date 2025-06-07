from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')

class WCTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, total_features=1000):
        self.total_features = total_features
        self.vocabulary_ = None
        self.vectorizer_ = None

    def fit(self, X, y):
        label_counts = Counter(y)
        total_docs = len(y)

        # Proportional feature allocation 
        class_feature_alloc = {
            label: int((count / total_docs) * self.total_features)
            for label, count in label_counts.items()
        }

        diff = self.total_features - sum(class_feature_alloc.values())
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for label, _ in sorted_labels[:abs(diff)]:
            class_feature_alloc[label] += int(np.sign(diff))

        global_vocab = set()

        # Vocabulary construction
        for label in sorted(label_counts, key=label_counts.get, reverse=True):
            n_features = class_feature_alloc[label]
            class_texts = [doc for doc, lab in zip(X, y) if lab == label]
            stop_words = list(global_vocab)

            tfidf = TfidfVectorizer(max_features=n_features, stop_words=stop_words)
            tfidf.fit(class_texts)

            vocab = set(tfidf.get_feature_names_out())
            global_vocab.update(vocab)

        self.vocabulary_ = list(global_vocab)

        # Final fit with the final vocabulary
        self.vectorizer_ = TfidfVectorizer(vocabulary=self.vocabulary_)
        self.vectorizer_.fit(X)
        
        return self

    def transform(self, X):
        return self.vectorizer_.transform(X)

class Word2VecTransformer:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, text_data):
        return np.array([
            np.mean([self.model.wv[word] for word in text.split() if word in self.model.wv], axis=0)
            if any(word in self.model.wv for word in text.split()) else np.zeros(self.model.vector_size)
            for text in text_data
        ])

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        return [self._process_summary(text, stop_words, stemmer) for text in X]

    def _process_summary(self, text, stop_words, stemmer):
        if not isinstance(text, str):
            return ''

         # 1. Lowercase
        text = text.lower()
    
        # 2. Remove special characters and punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text)
    
        # 3. Tokenize
        tokens = word_tokenize(text)
    
        # 5. Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]
    
        # 6. Apply stemming
        tokens = [stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)

def print_classification_report_matrix(y_test, y_pred):
    # Print basic metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-score (macro): {f1_score(y_test, y_pred, average='macro'):.3f}")
    print(f"F1-score (weighted): {f1_score(y_test, y_pred, average='weighted'):.3f}")

    # Print detailed classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Map labels to "Class 1", "Class 2", ...
    unique_labels = np.unique(y_test)
    unique_labels = sorted(unique_labels, key=lambda x: int(x.split()[0]))  # custom sorting if needed
    label_to_class = {label: f"Class {i+1}" for i, label in enumerate(unique_labels)}

    y_test_mapped = [label_to_class[label] for label in y_test]
    y_pred_mapped = [label_to_class[label] for label in y_pred]

    # Create confusion matrix
    conf_mat = confusion_matrix(y_test_mapped, y_pred_mapped, labels=list(label_to_class.values()))

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label_to_class.values()), yticklabels=list(label_to_class.values()))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

