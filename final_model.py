import re
from sklearn.base import BaseEstimator, TransformerMixin # type: ignore
from sklearn.pipeline import Pipeline   # type: ignore
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer  # type: ignore
from nltk.corpus import wordnet  # type: ignore
import nltk  # type: ignore
import sys
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.multiclass import OneVsRestClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score,  jaccard_score
import numpy as np
from joblib import dump, load


# Creation du pipeline de preproessing
class TagsCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.clean_tags(text) for text in X]

    def clean_tags(self, text):
        tags_to_transform = BeautifulSoup(text, "html.parser")
        return [tag.name for tag in tags_to_transform.find_all()]
    
    '''
    def transform(self, X, y=None):
        return [BeautifulSoup(text, "html.parser").get_text() for text in X]'''
    
class HtmlCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.clean_text(text) for text in X]

    def clean_text(self, text):
    # Delete HTML tags
        clean_text = re.sub(r'<.*?>', '', text)
    # Delete special characters and punctuation
        clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)
        return clean_text
    
    
    '''def transform(self, X, y=None):
        return [BeautifulSoup(text, "html.parser").get_text() for text in X]'''


class TextTokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [nltk.word_tokenize(text) for text in X]

class TextLower(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [[word.lower() for word in text] for text in X]
    
class TextStopWordRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        additional_stopwords = {"im", "ive",'new','like','get','want','would','need','use','using','used','one','two','three','four','five','six','seven','eight','nine','ten',
                                'public','like', 'doesnt','try','thanks','i',
                                'microsoftextensionsdependencyinjectionactivatorutilitiesgetserviceorcreateinstanceiserviceprovider',
                                'uscentraldockerpkgdevprojectidreponamev','use','gt','orgopenqaseleniuminteractionsactionsbuiltactionperformactionsjava',
                                'pageobjectsactivitypageobjectclickaudioinlinestopinactivitypageobjectjava','stepdefinitionactivitytheaudioplayeroftheelementisstoppedinactivityjava'}
        stop_words.update(additional_stopwords)
        return [[word for word in text if word not in stop_words] for text in X]
    
class TextLemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [[self.lemmatize(word) for word in text] for text in X]

    def lemmatize(self, word):
        pos_tag = self.get_wordnet_pos(word)
        if pos_tag:
            return self.lemmatizer.lemmatize(word, pos=pos_tag)
        else:
            return word

    def get_wordnet_pos(self, word):
        """Map POS tag to first character used by WordNetLemmatizer"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)
    
pipeline = Pipeline([
    ('html_cleaner', HtmlCleaner()),
    ('tokenizer', TextTokenizer()),
    ('text_lower', TextLower()),
    ('stop_words', TextStopWordRemover()),
    ('lemmatizer',TextLemmatizer()),
])

# Pipeline de nettoyage our les tags
pipeline_tags = Pipeline([('tags_cleaner', TagsCleaner())])

#Pipeline de vectorisation avec tfidf 
class CustomTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=0.03, max_df=0.007, ngram_range=(1, 1), norm='l2', use_idf=True, stop_words='english'):
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.norm = norm
        self.use_idf = use_idf
        self.stop_words = stop_words
        self.vectorizer = TfidfVectorizer(min_df=self.min_df, max_df=self.max_df,
                                          ngram_range=self.ngram_range, norm=self.norm,
                                          use_idf=self.use_idf, stop_words=self.stop_words)

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X, y=None):
        return self.vectorizer.transform(X)
    
def create_tfidf_pipeline():
    pipeline_tfidf = Pipeline([
        
        ("tfidf", TfidfVectorizer(min_df=0.007, max_df=0.9, ngram_range=(1, 1), stop_words='english')),
       # ("clf", OneVsRestClassifier(LogisticRegression())),
    ])
    return pipeline_tfidf

def create_clf_pipeline():
    pipeline_clf = Pipeline([
        ("clf", OneVsRestClassifier(LogisticRegression()))
    ])
    return pipeline_clf


def select_top_n_tags(probabilities, threshold, top_n=5):
    # Créer une matrice de zéros de la même forme que probabilities
    final_tags = np.zeros(probabilities.shape)
    
    for i in range(probabilities.shape[0]):
        # Trouver les indices qui satisfont le seuil
        eligible_indices = np.where(probabilities[i] >= threshold)[0]
        
        # Trier ces indices en fonction de leur probabilité
        top_indices = eligible_indices[np.argsort(probabilities[i][eligible_indices])][::-1]
        
        # Sélectionner les top N indices
        top_n_indices = top_indices[:top_n]
        
        # Mettre à 1 les positions correspondant aux tags sélectionnés
        final_tags[i, top_n_indices] = 1
    print("final tags debug:",final_tags)
    return final_tags

   


def preprocess_data(data):
    # create and instanciate de preprocessing Pipeline
    preprocessed_data = pipeline.transform(data)
    return preprocessed_data

def vectorize_data(data):
    # create and instanciate Vectorisation
    data = [' '.join(text) for text in data]
    vectorized_data = pipeline_tfidf.transform(data)
    return vectorized_data

def final_model(data):
    
    # Predict probabilities
    predictions = pipeline_clf.predict_proba(data)
    # Sélect tags
    probabilities = predictions
    binary_predictions = select_top_n_tags(probabilities, 0.2, top_n=7)
    
    predictions_labels =mlb.inverse_transform(binary_predictions)
    return predictions_labels


##### Preprocessing #####

# Load data
df = pd.read_csv('QueryResults2.csv')


# preprocessing de Body et Title
df['Body'] = df['Body'].fillna('')
df['Title'] = df['Title'].fillna('')

# Combinaison des colonnes avec une gestion propre
df['Comb'] = df.apply(lambda row: str(row['Body']) + " " + str(row['Title']), axis=1)
comb= []
comb = df.iloc[:].Comb[:3500]
preprocessed_comb = pipeline.fit_transform(comb)
print('preprocessed_comb \n',preprocessed_comb[:1], '\n')


# Enregistrer la fonction
dump(preprocess_data, 'preprocess_function.joblib')
# Charger la fonction
loaded_preprocess_function = load('preprocess_function.joblib')

# Utiliser la fonction chargée pour traiter de nouvelles données
new_data = ["android jetpack navigation bottomnavigationview youtube instagram proper back navigation fragment back "]
processed_new_data = loaded_preprocess_function(new_data)
print('processed_new_data',processed_new_data)

# preprocessing des tags
df['Tags'] = df.apply(lambda row: str(row['Tags']), axis =1)
tags = []
tags = df.iloc[:].Tags[:3500]
print('tags =',tags)
preprocessed_tags = pipeline_tags.fit_transform(tags)
print("preprossed tags",preprocessed_tags[:10])

# Filtrer les tags op 20 tags
'''from collections import defaultdict
def calculate_token_frequency(body):
    token_frequency = defaultdict(int)
    for text in body:
        for token in text:
            token_frequency[token] += 1
    sorted_token_frequency = dict(sorted(token_frequency.items(), key=lambda x: x[1], reverse=True))
    return sorted_token_frequency
tags_frequency = calculate_token_frequency(preprocessed_tags)

tags_to_keep = list(pd.Series(tags_frequency).sort_values(ascending=False).iloc[:15].index)
print('tags to keep',tags_to_keep)
#preprocessed_tags = df[df['Tags'].apply(lambda x: any(tag in tags_to_keep for tag in x.split()))]

# Filtrez les tags
preprocessed_tags = df[df['Tags'].apply(lambda tags: any(tag in tags_to_keep for tag in tags))]
print('preprocessed_tags',preprocessed_tags.head())'''
final_df = pd.DataFrame({'Comb':preprocessed_comb, 'Tags':preprocessed_tags})
print('final_df',final_df.head())

##### Transforamtion #####

X = final_df['Comb']
y = final_df['Tags']

X_train, X_test, y_train, y_test = train_test_split(X,y,
    test_size=0.2,
    random_state=42
)
#print('X_train',X_train[:10], y_train[:10])
print('X_train shape',X_train.shape,'X_train shape',X_test.shape)
X_train_sample = X_train.sample(2500, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]
X_test_sample = X_test.sample(50, random_state=42)
y_test_sample = y_test.loc[X_test_sample.index]

print('X_train shape',X_train_sample.shape,'X_train shape',X_test_sample.shape)
## Convertir y_train et y_test en matrice binaire
mlb = MultiLabelBinarizer()

# Adapter MultiLabelBinarizer et transformer les étiquettes
y_train_sample_trans = mlb.fit_transform(y_train_sample)
y_test_sample_trans = mlb.transform(y_test_sample) 
print('y_train_sample shape',y_train_sample_trans.shape)

pipeline_tfidf = create_tfidf_pipeline()

# Convertir les listes en chaînes de caractères
X_train_sample_str = [' '.join(text) for text in X_train_sample]
X_test_sample_str = [' '.join(text) for text in X_test_sample]

## transformation X en matrice tfidf
X_train_sample_trans = pipeline_tfidf.fit_transform(X_train_sample_str)
X_test_sample_trans = pipeline_tfidf.transform(X_test_sample_str)
print('X_train_sample_trans',X_train_sample_trans)
pipeline_clf = create_clf_pipeline()

pipeline_clf.fit(X_train_sample_trans, y_train_sample_trans)


predictions = pipeline_clf.predict_proba(X_test_sample_trans)
print("shape de prediction",predictions.shape)
print("shape de prediction",X_test_sample_trans.shape)

probabilities = predictions
binary_predictions = select_top_n_tags(probabilities, 0.15, top_n=7)

print("Precision:", precision_score(y_test_sample_trans, binary_predictions, average='samples'))
print("Recall:", recall_score(y_test_sample_trans, binary_predictions, average='samples'))
print("F1 Score:", f1_score(y_test_sample_trans, binary_predictions, average='samples'))
print("Jaccard Score:", jaccard_score(y_test_sample_trans, binary_predictions, average='samples'))

dump(vectorize_data, 'vectorize_function.joblib')
loaded_vectorize_function = load('vectorize_function.joblib')
new_data_vectorized = loaded_vectorize_function(processed_new_data)


dump(final_model, 'final_model.joblib')
loaded_final_model = load('final_model.joblib')
new_data_pred = loaded_final_model(new_data_vectorized)
#tag = mlb.inverse_transform(new_data_pred)
print('new_data_pred',new_data_pred)    