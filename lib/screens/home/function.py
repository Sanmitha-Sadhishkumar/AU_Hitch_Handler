import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import sklearn

def sample(text):
    comment ={'id':[1],'comment_text':[text]}

    comment = pd.DataFrame(comment)

    loaded_model = joblib.load('clf.joblib')

    result=loaded_model.predict(comment)
    
    if result[0]=='yes':
        return 'Comment is Toxic'
    else:
        return 'Comment is not Toxic'
