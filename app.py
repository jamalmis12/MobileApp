import onnxruntime as ort
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load your ONNX model
onnx_model = ort.InferenceSession("D:/JamXin/Model/best.onnx")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data (you would send input in JSON format from the client)
    data = request.get_json()
    input_data = np.array(data['input'])  # assuming 'input' is passed as an array/list
    
    # Preprocess the input data if necessary
    # Example: input_data = preprocess(input_data)

    # Perform inference
    outputs = onnx_model.run(None, {"input_name": input_data})  # 'input_name' must match the model input name
    
    # Return predictions in JSON format
    return jsonify({"prediction": outputs[0].tolist()})

if __name__ == "__main__":
    app.run(debug=True)
