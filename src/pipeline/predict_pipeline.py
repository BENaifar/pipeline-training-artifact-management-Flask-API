import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path="artifacts\model.pkl"
            preprocessor_path="artifacts\preprocessor.pkl"
            
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            
            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        PassengerId: int,
        Pclass: int,
        Name: str,
        Sex: str,
        Age: float,
        SibSp: int,
        Parch: int,
        Ticket: str,
        Fare: float,
        Cabin: str,
        Embarked: str
    ):
        self.PassengerId = PassengerId
        self.Pclass = Pclass
        self.Name = Name
        self.Sex = Sex
        self.Age = Age
        self.SibSp = SibSp
        self.Parch = Parch
        self.Ticket = Ticket
        self.Fare = Fare
        self.Cabin = Cabin
        self.Embarked = Embarked

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "PassengerId": [self.PassengerId],
                "Pclass": [self.Pclass],
                "Name": [self.Name],
                "Sex": [self.Sex],
                "Age": [self.Age],
                "SibSp": [self.SibSp],
                "Parch": [self.Parch],
                "Ticket": [self.Ticket],
                "Fare": [self.Fare],
                "Cabin": [self.Cabin],
                "Embarked": [self.Embarked],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)