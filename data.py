import pandas as pd
pd.set_option("display.max_columns", None)
import os
import re
import string

#%%

class DataRepository:
    def __init__(self):
        pass
    
    #%% DATA LOADING
    
    def load_data(self):
        file_list = os.listdir("data")
        result = pd.DataFrame()
        
        for file_name in file_list:
            #---
            file_path = "data/" + file_name
            df = pd.read_csv(file_path)
            
            # transform name => category + topic
            name_split = file_name.split(".")[0].split("_")
            cat = name_split[0]
            topic = "_".join(name_split[1:])
            df['cat'] = cat
            df['topic'] = topic
            
            #---
            result = pd.concat([result, df])
    
        result = result.reset_index(drop=True)
        return self.clean_data(result)


    #%% DATA CLEANING (SUPPORT FUNCTION)
    
    def clean_data(self, df):
        # 1. make lowercase
        df['content'] = df['content'].apply(lambda x: x.lower())

        # 2. remove punctuation, new line char, ... char, digit
        remove_str = "[" + re.escape(string.punctuation) + "\n" + "…" + "]+"  + "|" + r"\d+"

        df['content'] = df['content'].apply(
            lambda x: re.sub(remove_str, " ", x)
            )

        # clean up multiple space
        df['content'] = df['content'].apply(
            lambda x: re.sub(r"\s+", " ", x).strip()
            )

        # 3. tokenization
        df['content'] = df['content'].apply(
            lambda x: x.split(" ")
            )
        
        return df




