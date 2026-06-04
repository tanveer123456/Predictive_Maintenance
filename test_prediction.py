from src.pipeline.prediction_pipeline import (
    PredictPipeline,
    CustomData
)

data = CustomData(
    Type="L",
    Air_temperature=300,
    Process_temperature=310,
    Rotational_speed=3000,
    Torque=80,
    Tool_wear=250
)

pred_df = data.get_data_as_dataframe()

print("Input Data:")
print(pred_df)

predict_pipeline = PredictPipeline()

result = predict_pipeline.predict(pred_df)

print("\nPrediction:")
print(result)

if result[0] == 1:
    print("Machine Failure Predicted")
else:
    print("No Machine Failure Predicted")