
import re, nltk, time
import gensim, csv, pickle

from pprint import pprint
from multiprocessing import cpu_count
import pandas as pd
import dask.dataframe as dd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from gensim.models.doc2vec import LabeledSentence, Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn import linear_model

stop_words = set(stopwords.words('english'))

tags = {'news':1, 
        'clinicalTrials':2, 
        'conferences':3, 
        'profile':4, 
        'forum':5, 
        'publication':6, 
        'thesis': 7, 
        'guidelines':8, 
        'others':9
        }

split_per = 0.75        # Training-Test split ratio



def extract_tags(filename):
    '''
    Extract Tag ID and its corresponding Webpage ID for each example in training data.
    Print the table into a .csv file for reference
    '''
    df = pd.read_csv(filename)
    tags_id = []
    for row in df.itertuples(index=True, name='Pandas'):
       tags_id.append((row[1], tags[row[4]]))

    tags_df = pd.DataFrame.from_records(tags_id)
    with open('tags_id.csv', 'w') as fd:
        tags_df.to_csv(fd, header=False, index=False)
    tags_id = dict(tags_id)

    return tags_id


def preprocess(filename, tags_id):
    '''
    Preprocess the Html data in train.csv and store it into a csv file with ID and Label
    '''
    chunksize = 1000
    columns=['Webpage_id','Html', 'Tags']
    #html_df = pd.DataFrame(columns=columns)
    
    n = 0
    with open('html_tags.csv', 'w') as fd:
        wr = csv.writer(fd)
        wr.writerow(columns)
    
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        html = []

        for row in chunk.itertuples(index=True, name='Pandas'):
            #mystring = BeautifulSoup(row[2], 'html.parser').get_text()
            mystring = re.sub('[^A-Za-z]+', ' ', row[2])
            
            text = ' '.join([word.lower() for word in mystring.split() if word not in stop_words and len(word)>=2])
            if row[1] in tags_id:
                html.append((row[1], text, tags_id[row[1]]))
            else:
                html.append((row[1], text, ''))

        chunk_df = pd.DataFrame.from_records(html, columns=columns)
        
        with open('html_tags.csv', 'a') as fd:
            chunk_df.to_csv(fd, header=False, index=False)
        n+=1
        print(n)
        

def get_tag_data(filename, tags_id):
    '''
    Extract Html data and corresponding labels for training
    '''
    tags = []
    documents = []

    df = pd.read_csv(filename)
    df = df.sample(frac=1)            # randomize the dataframe

    for row in df.itertuples(index=True, name='Pandas'):
        if row[1] in tags_id:
            tags.append(int(row[3]))
            documents.append(row[2])
    return tags, documents


def get_test_tag_data(htmlfile, testfile):
    '''
    Extract Html data of the held out test dataset for prediction
    '''
    documents = []
    web_id = []
    test_df = pd.read_csv(testfile)

    web_id = test_df['Webpage_id']
    html_df = pd.read_csv(htmlfile)

    for idno in web_id:
       row = html_df.loc[html_df['Webpage_id']==idno, 'Html'].tolist()[0]
       documents.append(row)

    return web_id, documents


def test_data_prediction(X_new, ID_test):
    '''
    Predict the class labels for held out test data and print into a csv file
    '''
    #Load the vectors & classifier

    #tfidf_vec = pickle.load(open("tfidf_feature.pkl", "rb"))
    X_new_vec = vectorizer.transform(X_new)

    #lg_model = pickle.load(open("logreg_model.pkl","rb"))
    pred = logreg.predict(X_new_vec)

    pred_tag = []
    for i in pred:
        for tag, id_no in tags.iteritems():
            if id_no == i:
                pred_tag.append(tag)

    predictions = []
    columns=['Webpage_id', 'Tag']

    with open('submission.csv', 'w') as fd:
            wr = csv.writer(fd)
            wr.writerow(columns)

    for i in range(len(ID_test)):
        predictions.append((ID_test[i], pred_tag[i]))

    pred_df = pd.DataFrame.from_records(predictions, columns=columns)
    with open('submission.csv', 'a') as fd:
        pred_df.to_csv(fd, header=False, index=False)



print('Data processing')
start = time.time()

training_tag_id = extract_tags('train.csv')
preprocess('html_text.csv', training_tag_id)
Y, X = get_tag_data('html_tags.csv', training_tag_id)

end = time.time()
print(end-start)

num_ex = len(Y)
split = int(split_per * num_ex)



print('TF-IDF embeddings')
start = time.time()

vectorizer = TfidfVectorizer(max_features=5000)
tf_idf_matrix = vectorizer.fit_transform(X)
pickle.dump(tf_idf_matrix, open("tfidf_feature.pkl","wb"))

end = time.time()
print(end-start)



print('Logistic Regression model training')
start = time.time()

X_train = tf_idf_matrix[:split]
X_test = tf_idf_matrix[split:]
Y_train, Y_test = Y[:split], Y[split:]

logreg = linear_model.LogisticRegression(C=1e5)         # C = Regularization coefficient
logreg.fit(X_train,Y_train)
pickle.dump(logreg, open("logreg_model.pkl","wb"))

end = time.time()
print(end-start)



print('Prediction')
pred = logreg.predict(X_test)
accuracy = f1_score(Y_test, pred, average='weighted')   # Weighted F1 score
print("Weighted F1 score: %s" % accuracy)



print('Test Data processing')
start = time.time()

ID_test, X_new = get_test_tag_data('html_tags.csv', 'test.csv')

end = time.time()
print(end-start)


print('Predict Labels')
start = time.time()

test_data_prediction(X_new, ID_test)

end = time.time()
print(end-start)



'''
Results-

Held out Test data points = 25787
Training data points = 53447
Total data points = 79234

Time in seconds:
Data processing
12.7180430889
TF-IDF embeddings
149.059910059
Logistic Regression model training
332.070147038

Prediction
Weighted F1 score: 0.944854627028

Test Data processing
29.8206820488
Predict Labels
53.4433751106
'''



'''
Doc2Vec model was tried to be trained but Memory and Computation resources were not sufficient.
But, it can be trained on Cloud servers. Below is the code for doc2vec model training.

def doc2vec():
    '''
    #Reads entire training Html data and trains doc2vec model on it
    '''
    model = Doc2Vec(vector_size=50, 
                    min_count=2, 
                    epochs=5, 
                    window=3,
                    dm=1,
                    hs=0,
                    negative=5,
                    workers=cpu_count())
    
    chunksize = 1000
    html_text = []
    count = 0

    for chunk in pd.read_csv('html_text.csv', chunksize=chunksize):
        for row in chunk.itertuples(index=True, name='Pandas'):
            text = re.sub('[^A-Za-z]+', ' ', row[2])
            text = TaggedDocument(gensim.utils.simple_preprocess(text), [row[1]])
            html_text.append(text)
        count += 1
        print(count)

    print(len(html_text))

    model.build_vocab(html_text)
    model.train(html_text, total_examples=model.corpus_count, epochs=model.epochs)
    
    model.save('html_model')
    #model = Doc2Vec.load("tmpmodel")


def doc2vec_online():
    '''
    #Trains doc2vec vectors on Html data in online fashion with data read in batches
    '''
    chunksize = 10
    model = Doc2Vec(vector_size=50, 
                    min_count=2, 
                    epochs=5, 
                    window=3,
                    dm=1,
                    hs=0,
                    negative=5,
                    workers=cpu_count())
    n=0

    for chunk in pd.read_csv('html_data.csv', chunksize=chunksize):
        text_chunk = []
        for row in chunk.itertuples(index=True, name='Pandas'):
            text = BeautifulSoup(row[2], 'html.parser').get_text()
            text = re.sub('[^A-Za-z]+', ' ', text)

            html_text = TaggedDocument(gensim.utils.simple_preprocess(text), [row[1]])
            text_chunk.append(html_text)
        #print(text_chunk)
        
        if n==0:
            model.build_vocab(text_chunk)
            #model.save('tmpmodel')
            #model = Doc2Vec.load("tmpmodel")
        else:
            model.build_vocab(text_chunk, update=True)      
                    # update=True let's model train with data coming in batches
            #model.save('tmpmodel')
            #model = Doc2Vec.load("tmpmodel")

        model.train(text_chunk, total_examples=model.corpus_count, epochs=model.iter)
        n += 1
        print(n)

#doc2vec()
'''