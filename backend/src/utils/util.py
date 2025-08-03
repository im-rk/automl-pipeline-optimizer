import os
import sys
import pickle
from src.utils.exception import CustomException 

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(dir_path,'wb') as fp:
            pickle.dump(obj,fp)
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as fp:
            pickle.load(fp)
    except Exception as e:
        raise CustomException(e,sys)