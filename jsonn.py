#this code will diractly deploy in the AWS lambda funtion to create the url getway
import json

from sklearn.ensemble import IsolationForest

import numpy as np

   

def lambda_handler(event, context):

    # Check if the request contains the required fields

    if 'dataSet' not in event:

        return {

            'body': json.dumps({'error': 'Invalid request. Missing dataSet field.'}),

            'statusCode': 400

        }

   

    response_anomalies = []

   

    # Process each data set in the payload

    for data_item in event['dataSet']:

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

                response_anomalies.append({'dpCode': data_item.get('dpCode', ''), 'error': f'Value {current_value} is not present in the given data.'})

                continue




            # Create an Isolation Forest model and fit it to the data

            clf = IsolationForest(contamination=0.5).fit(data)




            # Predict the anomalies

            anomalies = clf.predict(data)




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

            response_anomalies.append({'dpCode': data_item.get('dpCode', ''), 'error': 'An error occurred. ' + str(e)})




    # Return the response with all anomalies

    return {

        'body': json.loads(json.dumps(response_anomalies)),

        'statusCode': 200

    }