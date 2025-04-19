import pandas as pd
pd.set_option("display.max_columns", None)
import os

#%% DATA LOADING

def load_data():
    file_list = os.listdir("data")
    result = pd.DataFrame()
    
    for file_name in file_list:
        file_path = "data/" + file_name
        df = pd.read_csv(file_path)

        name_split = file_name.split(".")[0].split("_")
        cat = name_split[0]
        topic = "_".join(name_split[1:])

        df['cat'] = cat
        df['topic'] = topic
        
        result = pd.concat([result, df])

    result = result.reset_index(drop=True)
    return result

df = load_data()

#%% DATA CLEANING