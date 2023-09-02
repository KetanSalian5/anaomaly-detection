from flask import Flask, request, jsonify
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

app = Flask(__name__)

@app.route('/anomaly', methods=['POST'])
def predict():
    # Check if the request contains the required fields
    if 'dataSet' not in request.json:
        return jsonify({'error': 'Invalid request. Missing dataSet field.'}), 400
    
    # Get the data sets from the request
    data_sets = request.json['dataSet']
    
    try:
        anomaly_data_list = []
        
        for data_item in data_sets:
            # Check if the data item contains the required fields
            if 'data' not in data_item or 'currentValue' not in data_item:
                anomaly_data_list.append({'error': 'Invalid data item. Missing data or currentValue field.'})
                continue
            
            # Get the data and current value from the data item
            data_str = data_item['data']
            current_value = float(data_item['currentValue'])
            
            # Split the string, remove the double quotes, and convert to float
            data = np.array([float(val.strip('"')) for val in data_str]).reshape(-1, 1)
            
            # Check if the number of values is less than two
            if data.shape[0] < 3:
                anomaly_data_list.append({'error': 'Invalid data. At least one more values are required.'})
                continue

            # Create a One-Class SVM model and fit it to the data
            clf = LocalOutlierFactor().fit(data)

            # Predict the anomalies
            anomalies = clf.predict(data)

            # Convert the anomalies to boolean values
            anomaly_data = data[anomalies == -1].flatten()

            # Convert the anomaly data to a list of strings instead of decimal value.
            anomaly_data_str = [str(int(val)) if val.is_integer() else str(val) for val in anomaly_data]

            # Prepare the response for the current data item
            response = {    
                'dpCode': data_item.get('dpCode', ''),
                'anomalyData': anomaly_data_str
            }
            
            anomaly_data_list.append(response)

        return jsonify(anomaly_data_list), 200
    
    except Exception as e:
        return jsonify({'error': 'An error occurred. ' + str(e)}), 500


@app.route('/anomaly/check', methods=['POST'])
def check_anomaly():
    # Check if the request contains the required fields
    if 'dataSet' not in request.json:
        return jsonify({'error': 'Invalid request. Missing dataSet field.'}), 400

    response_anomalies = []

    # Process each data set in the payload
    for data_item in request.json['dataSet']:
        # Check if the data item contains the required fields
        if 'data' not in data_item or 'currentValue' not in data_item:
            response_anomalies.append({'error': 'Invalid data item. Missing data or currentValue field.'})
            continue

        try:
            # Get the data and current value from the data item
            data_str = data_item['data']
            current_value = float(data_item['currentValue'])

            # Split the string, remove the double quotes, and convert to float
            data = np.array([float(val.strip('"')) for val in data_str]).reshape(-1, 1)

            # Check if the current value is present in the data set
            if current_value not in data:
                response_anomalies.append({'dpCode': data_item.get('dpCode', ''),
                                           'error': f'Value {current_value} is not present in the given data.'})
                continue

            # Create a Local Outlier Factor model and fit it to the data
            lof = LocalOutlierFactor(contamination=0.5).fit(data)

            # Predict the anomalies
            anomalies = lof.predict(data)

            # Check if the current value is an anomaly
            is_anomaly = current_value in data[anomalies == -1]

            # Calculate the mean and median
            mean = round(np.mean(data), 2)
            median = np.median(data)

            # Prepare the response for the current data item
            response_anomaly = {
                'isAnomaly': is_anomaly,
                'mean': mean,
                'median': median,
                'dpCode': data_item.get('dpCode', '')
            }

            response_anomalies.append(response_anomaly)

        except Exception as e:
            response_anomalies.append({'dpCode': data_item.get('dpCode', ''),
                                       'error': 'An error occurred. ' + str(e)})

    # Return the response with all anomalies
    return jsonify(response_anomalies), 200


if __name__ == '__main__':
    app.run()
