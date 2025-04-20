from data import DataRepository
from naive_bayes import NaiveBayesClassifier
import numpy as np
import pandas as pd

#%% NEW VERSION
# Load data
repo = DataRepository()
df = repo.load_data()

# Train, Test data
df_train = df[df['cat']=='training']
df_test = df[df['cat']=='testing']

# Load Model
model = NaiveBayesClassifier()

#%% TESTING
model.fit(df_train['content'].values, df_train['topic'].values)

pred = []
for content in df_test['content']:
    pred.append(model.predict(content))
    
df_test['pred'] = pred

acc = (df_test['topic'] == df_test['pred']).sum() / len(df_test)
        
print(f"Accuracy: {acc:.0%}")

#%% PREP (OLD VERSION)
 
# Load data
repo = DataRepository()
df = repo.load_data()

# Train, Test data
df_train = df[df['cat']=='training']
df_test = df[df['cat']=='testing']

categories = df_train['topic'].unique()

# Get size of vocabulary
def get_vocab(ds):
    vocab = []
    for x in ds:
        vocab.extend(x)
        
    return vocab

def get_vocab_size(vocab, bank_type=None):
    if bank_type == "unique":
        vocab = set(vocab)
    
    return len(vocab)

# Get word count in a list
def count_word(vocab, vocab_train=None):
    
    word_count = {}
    if vocab_train:
        for word in set(vocab_train):
            word_count[word] = 0
    else:
        for word in set(vocab):
            word_count[word] = 0
        
    for word in vocab:
        word_count[word] += 1
        
    return word_count

# Estimate word prob (Step 3)
def cal_word_prob(Nci, Nc, V_size):
    alpha = 1  # As Laplace smoothing
    P_word_category = (Nci + alpha) / (Nc + alpha * V_size)
    
    return P_word_category

#%% TRAINING
# Step 1. Build vocabulary
vocab_train = get_vocab(df_train['content'])
V_size = get_vocab_size(vocab_train, bank_type="unique")

report = {cat: {} for cat in categories}

posterior_cat = {cat: [] for cat in categories}

for cat in categories:
    report[cat]['vocab'] = get_vocab(df_train[df_train['topic']==cat]['content'])
    report[cat]['count'] = count_word(report[cat]['vocab'], vocab_train)
    
# Step 2. Total word in each catetory
    report[cat]['Nc'] = get_vocab_size(report[cat]['vocab'])
    
# Step 4. Prior probability of category
    report[cat]['prior'] = (
        len(df_train[df_train['topic']==cat])/len(df_train)
         )
    
#%% TESTING
    # Step 5. Test data with word count
    
    for testing_vocab in df_test['content']:
    
        testing_count = count_word(testing_vocab)
    
        # starting for posterior probability
        posterior = report[cat]['prior']
        
        for word in testing_count.keys():
            if word in set(vocab_train):
                Nci = report[cat]['count'][word]
            else:
                Nci = 0
                
            Nc = report[cat]['Nc']
            
            # update posterior    
            posterior *= (cal_word_prob(Nci, Nc, V_size) ** testing_count[word])
         
        log_posterior = np.log(posterior)
        
        posterior_cat[cat].append(log_posterior)

pred = []

for n in range(len(df_test)):
    # initialization of argmax value
    argmax = categories[0]
    
    for cat in categories:
        if posterior_cat[cat][n] > posterior_cat[argmax][n]:
            argmax = cat
            
    pred.append(argmax)
    
df_test['pred'] = pred

acc = (df_test['topic'] == df_test['pred']).sum() / len(df_test)
        
print(f"Accuracy: {acc:.0%}")
