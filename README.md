# Innoplexus-Online-Hackathon

It contains the datasets and my submission code for 'Innoplexus Online Hackathon 2018'. Read the 'Problem statement.pdf' to understand the challenge.

Link to the challenge: https://datahack.analyticsvidhya.com/contest/innoplexus-online-hiring-hackathon-ai-challenge/

*Dataset size* - 
Held out Test data points = 25787, Training data points = 53447, Total data points = 79234

Word vectorization model used: TF-IDF, Classification model used: Logistic Regression

Doc2Vec model was tried for word vectorization on the text data, but Memory and Computation resources were not sufficient. However, it can be trained on Cloud servers. The code for doc2vec model training is commented in the python code file.

Weighted F1 accuracy score: 0.944854627028
