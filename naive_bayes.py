import numpy as np


class NaiveBayesClassifier:
    #%% INIT
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # 1 for Laplace 
        
    #%% 
    def fit(self, X: list[list[str]], C: list[str]) -> None:
        """Learn:
            - general vocab & vocab size
            - Nc (each class)
            - priors (each class)
        """
        ### 0. Result placement
        class_trained = {
            c: {
                'vocab': [],
                'count': {},
                'Nc': 0,
                'prior': 0
                } for c in C
            }
        
        ### 1. Build general vocab
        vocab_train = self.get_vocab(X, nested=True)
        V_size = len(set(vocab_train))
        
        # Create vocab of each class
        for x, c in zip(X, C):
            class_trained[c]['vocab'].extend(self.get_vocab(x, nested=False))
            
            ### 4. Compute class prior
            class_trained[c]['prior'] += 1/len(C) 
        
        
        for c in set(C):
            # Count word
            class_trained[c]['count'] = self.count_word(class_trained[c]['vocab'])
            
            ### 2. Count feature frequency
            class_trained[c]['Nc'] = len(class_trained[c]['vocab'])
        
        self.class_trained_ = class_trained
        self.V_size_ = V_size
        
    # Support Function    
    def get_vocab(self, 
                  X: list[list[str]] | list[str],
                  nested: bool = True
                  ) -> list[str]:
        
        vocab = []
        if nested:
            for x in X:
                vocab.extend(x)
        else:
            for x in X:
                vocab.append(x)
            
        return vocab
    
    def count_word(self, vocab: list[str]) -> dict[str, int]:
        
        word_count = {}
        unique_vocab = set(vocab)
        
        for word in unique_vocab:
            word_count[word] = 0
        
        for word in vocab:
            word_count[word] += 1
            
        return word_count
          
    #%%
    def predict_one(self, X_test: list[str]) -> list[str]:
        # Get list of class from training data
        C = self.class_trained_.keys()
        
        # Initiate posterior with prior value
        posterior_log_proba = {
            c: np.log(self.class_trained_[c]['prior']) for c in C
            }
        
        # Create vocab of testing item
        x_vocab = self.get_vocab(X_test, nested = False)  # -> list
        
        for word in x_vocab:
            for c in C:
                Nci = self.class_trained_[c]['count'].get(word, 0)
                
                posterior_log_proba[c] += np.log(
                    (Nci + self.alpha) / 
                    (self.class_trained_[c]['Nc'] + self.alpha * self.V_size_)
                    )
                
        return max(posterior_log_proba, key = posterior_log_proba.get)
        
        
