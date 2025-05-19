from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)

mlflow.set_tracking_uri(
    "http://localhost:9090"
)  # cambiar en función de su servidor  http://127.0.0.1:5050
model = mlflow.sklearn.load_model(
    "models:/LR Model Prediction/1"
)  # cambiar en función de su modelo models:/sentimientos_texto_model/3


@app.route("/predict", methods=["GET"])
def predict_get():
    word_count = int(request.args.get("word_count", 0))
    sum_value = int(request.args.get("sum", 0))

    df = pd.DataFrame([{"Word count": word_count, "Sum": sum_value}])
    prediction = model.predict(df)
    return jsonify({"predicted_shares": int(prediction[0])})


if __name__ == "__main__":
    app.run(port=5000)
