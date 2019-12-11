import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,label_ranking_average_precision_score, fbeta_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from scipy.stats.mstats import gmean
import pickle
nltk.download(['wordnet','punkt','stopwords'])



def load_data(database_filepath):
    '''
    Function to load data from the Database
    
    Args : 
        database_filepath : relative path of the database 
        
    Returns :
        The features list X,y and the category names of features in y
    '''
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    #engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('disasterTable', con=engine)
    categories = df.columns[4:]
    X = df[['message']].values[:, 0]
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names



def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    lemmatizer=WordNetLemmatizer()
    # Detecte URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    # Normalize and tokenize
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def build_model(X,Y):
    '''
    Creates and returns the pipeline(model)
    
    Args:
        X,Y
    '''
    vect = CountVectorizer(tokenizer=tokenize)
    X_vectorized = vect.fit_transform(X)
    keys, values = [], []
    for k, v in vect.vocabulary_.items():
        keys.append(k)
        values.append(v)
    vocabulary = pd.DataFrame.from_dict({'words': keys, 'counts': values})
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model and returns the score 
    
    Args: 
        model : trained model
        X_test,Y_test and the category name of target variable
    '''
    y_pred = model.predict(X_test)
    multi_f1_score = multioutput_f1score(Y_test,y_pred, beta = 1)
    overall_acc = (y_pred == Y_test).mean().mean()
    print('Average Accuracy {0:.2f}% \n'.format(overall_acc*100))
    print('F1 score {0:.2f}%\n'.format(multi_f1_score*100))
 

def multioutput_f1score(y_test,y_pred,beta=1):
    '''
    Calculates the f1 score for model
    
    Args:
        y_test and y_pred
    '''
    score_lst = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    if isinstance(y_test, pd.DataFrame) == True:
        y_test = y_test.values
    for column in range(0,y_test.shape[1]):
        score = fbeta_score(y_test[:,column],y_pred[:,column],beta,average='weighted')
        score_lst.append(score)
    f1_score_numpy = np.asarray(score_lst)
    f1_score_numpy = f1_score_numpy[f1_score_numpy<1]
    f1_score_ = gmean(f1_score_numpy)
    return  f1_score_


def save_model(model, model_filepath):
    '''
    Saves the model as the name passed to this function
    
    Args:
        model : trained model
        model_filepath : name of the file (better if passed with .pkl extension)
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X,Y)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()