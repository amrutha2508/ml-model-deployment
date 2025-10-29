import json
import requests

data = [
        [1, 3, 1, 4],
        [5, 2, 3, 1]
    ]

url = "http://localhost:8000/predict/"

predictions = []

for record in data:
    payload = {'features': record}
    payload = json.dumps(payload)
    response = requests.post(url,payload)
    predictions.append(response.json()['predicted_class'])

print(predictions)