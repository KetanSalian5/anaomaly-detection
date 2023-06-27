from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
import numpy as np

app = Flask(__name__)

@app.route('/anomaly', methods=['POST'])
def predict():
    # Check if the request contains the required fields
    if 'data' not in request.json:
        return jsonify({'error': 'Invalid request. Missing data field.'}), 400
    
    # Get the data from the request
    data_str = request.json['data']
    
    try:
        # Split the string, remove the double quotes, and convert to float
        data = np.array([float(val.strip('"')) for val in data_str]).reshape(-1, 1)
        
        # Check if the number of values is less than two
        if data.shape[0] < 3:
            return jsonify({'error': 'Invalid data. At least one more values are required.'}), 400

        # Create an Isolation Forest model and fit it to the data
        clf = IsolationForest(contamination='auto').fit(data)

        # Predict the anomalies
        anomalies = clf.predict(data)

        # Convert the anomalies to boolean values
        anomaly_data = data[anomalies == -1].flatten()

        # so Convert the anomaly data to a list of strings instead of decimal value.
        anomaly_data_str = [str(int(val)) if val.is_integer() else str(val) for val in anomaly_data]

        # Prepare the response
        response = {
            'anomalyData': anomaly_data_str
        }

        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': 'An error occurred. ' + str(e)}), 500
@app.route('/anomaly/check', methods=['POST'])
def check_anomaly():
    # Check if the request contains the required fields
    if 'data' not in request.json or 'currentValue' not in request.json:
        return jsonify({'error': 'Unable to perform anomaly detection as minimum 3 values required!'}), 400
    
    # Get the data and current value from the request
    data_str = request.json['data']
    current_value = float(request.json['currentValue'])
    
    try:
        # Split the string, remove the double quotes, and convert to float
        data = np.array([float(val.strip('"')) for val in data_str]).reshape(-1, 1)

        # Check if the current value is present in the data set
        if current_value not in data:
            
            return jsonify({'error': f'Value {current_value} is not present in the given data.',}), 400

        # Create an Isolation Forest model and fit it to the data
        clf = IsolationForest(contamination='auto').fit(data)

        # Predict the anomalies
        anomalies = clf.predict(data)

        # Check if the current value is an anomaly
        is_anomaly = current_value in data[anomalies == -1]

        # Prepare the response
        response = {
            'isAnomaly': is_anomaly
        }

        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': 'An error occurred. ' + str(e)}), 500


if __name__ == '__main__':
    app.run()
