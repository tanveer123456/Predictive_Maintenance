
from flask import Flask, request, render_template
from datetime import datetime

from src.pipeline.prediction_pipeline import (
    PredictPipeline,
    CustomData
)

application = Flask(__name__)
app = application


@app.route("/")
def home():

    return render_template(
        "home.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = CustomData(

            Type=request.form.get("Type"),

            Air_temperature=float(
                request.form.get(
                    "Air_temperature"
                )
            ),

            Process_temperature=float(
                request.form.get(
                    "Process_temperature"
                )
            ),

            Rotational_speed=float(
                request.form.get(
                    "Rotational_speed"
                )
            ),

            Torque=float(
                request.form.get(
                    "Torque"
                )
            ),

            Tool_wear=float(
                request.form.get(
                    "Tool_wear"
                )
            )
        )

        # Input Validation

        if not (
            295 <= data.Air_temperature <= 305
        ):
            raise ValueError(
                "Air Temperature must be between 295 and 305 K"
            )

        if not (
            305 <= data.Process_temperature <= 315
        ):
            raise ValueError(
                "Process Temperature must be between 305 and 315 K"
            )

        if not (
            1100 <= data.Rotational_speed <= 3000
        ):
            raise ValueError(
                "Rotational Speed must be between 1100 and 3000 RPM"
            )

        if not (
            3 <= data.Torque <= 80
        ):
            raise ValueError(
                "Torque must be between 3 and 80 Nm"
            )

        if not (
            0 <= data.Tool_wear <= 250
        ):
            raise ValueError(
                "Tool Wear must be between 0 and 250 Minutes"
            )

        df = data.get_data_as_dataframe()

        predictor = PredictPipeline()

        prediction, probability = (
            predictor.predict(df)
        )

        probability_percent = round(
            probability * 100,
            2
        )

        health_score = round(
            100 - probability_percent,
            2
        )

        confidence_score = round(
            max(
                probability_percent,
                100 - probability_percent
            ),
            2
        )

        prediction_time = (
            datetime.now().strftime(
                "%d %b %Y %I:%M:%S %p"
            )
        )

        # Model Information

        model_name = (
            "Gradient Boosting"
        )

        roc_auc = 96.88

        model_f1_score = 70.27

        # Risk Assessment

        if probability_percent < 30:

            risk = "LOW"

            status = "HEALTHY"

            alert_class = (
                "alert-success"
            )

            recommendation = (
                "Machine health is stable. Continue regular monitoring."
            )

        elif probability_percent < 70:

            risk = "MEDIUM"

            status = "WARNING"

            alert_class = (
                "alert-warning"
            )

            recommendation = (
                "Schedule preventive maintenance soon."
            )

        else:

            risk = "HIGH"

            status = "CRITICAL"

            alert_class = (
                "alert-danger"
            )

            recommendation = (
                "Immediate maintenance required. High failure probability detected."
            )

        # Prediction Result

        if prediction == 1:

            result = (
                "Machine Failure Predicted"
            )

        else:

            result = (
                "No Machine Failure Predicted"
            )

        return render_template(

            "home.html",

            prediction_text=result,

            probability=probability_percent,

            confidence_score=confidence_score,

            prediction_time=prediction_time,

            health_score=health_score,

            risk=risk,

            status=status,

            recommendation=recommendation,

            alert_class=alert_class,

            model_name=model_name,

            roc_auc=roc_auc,

            f1_score=model_f1_score,

            machine_type=data.Type,

            air_temp=data.Air_temperature,

            process_temp=data.Process_temperature,

            rpm=data.Rotational_speed,

            torque=data.Torque,

            tool_wear=data.Tool_wear

        )

    except Exception as e:

        return render_template(
            "home.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

