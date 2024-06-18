import re
import joblib
import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
#Задание некоторых обработчиков текста
nlp = spacy.load('en_core_web_sm')
stop_words_en = set(stopwords.words('english'))
stop_words_ru = set(stopwords.words('russian'))
stemmer_ru = SnowballStemmer("russian")
#предварительная обработка текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n', ' ', text)
    tokens = text.split()
    processed_tokens = []
    for token in tokens:
        if token in stop_words_en or token in stop_words_ru:
            continue
        if re.match(r'[a-zA-Z]', token):
            # Лемматизация для английских слов
            doc = nlp(token)
            token = doc[0].lemma_
        else:
            # Стемминг для русских слов
            token = stemmer_ru.stem(token)
        processed_tokens.append(token)

    return ' '.join(processed_tokens)
#Загрузка данных из файлов
def load_data(text_file, labels_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = [int(label.strip()) for label in f]
    return texts, labels

text_file = 'TF-IDF data.txt'
labels_file = 'TF-IDF data labels.txt'
texts, labels = load_data(text_file, labels_file)
texts_preprocessed = [preprocess_text(text) for text in texts]
X_train, X_test, y_train, y_test = train_test_split(texts_preprocessed,
    labels, test_size=0.3, random_state=12)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

svm_classifier = SVC(kernel='linear', probability=True)
rf_classifier = RandomForestClassifier(n_estimators=100)

voting_classifier = VotingClassifier(estimators=[('svm', svm_classifier)], voting='soft')
voting_classifier.fit(X_train_tfidf, y_train)
predictions = voting_classifier.predict(X_test_tfidf)

# Выведите отчет о классификации
print("Отчет о классификации для ансамбля моделей:")
print(classification_report(y_test, predictions))
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(voting_classifier, 'classifier.pkl')
