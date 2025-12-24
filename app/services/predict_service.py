from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def predict_passenger(data: dict) -> str:
    passenger_data = CustomData(
        PassengerId=2,  # dummy ID, can be auto-generated
        Pclass=int(data.get("Pclass")),
        Name=data.get("Name"),
        Sex=data.get("Sex"),
        Age=float(data.get("Age")) if data.get("Age") else None,
        SibSp=int(data.get("SibSp")),
        Parch=int(data.get("Parch")),
        Ticket=data.get("Ticket"),
        Fare=float(data.get("Fare")),
        Cabin=data.get("Cabin"),
        Embarked=data.get("Embarked")
    )
    
    pred_df = passenger_data.get_data_as_data_frame()
    pipeline = PredictPipeline()
    results = pipeline.predict(pred_df)
    return "Survived" if int(results[0]) == 1 else "Did not survive"
