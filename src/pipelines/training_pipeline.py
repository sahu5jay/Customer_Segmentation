import os 
from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
# from src.components.model_training import ModelTrainer

if __name__ == '__main__':
    obj = DataIngestion()
    sub,sub2 = obj.initiate_data_ingestion()

    # obj2 = DataTransformation()
    # train_arr,test_arr,_=obj2.initate_data_transformation(sub,sub2)

    # obj3 = ModelTrainer()
    # obj3.iniate_model_training(train_arr,test_arr)