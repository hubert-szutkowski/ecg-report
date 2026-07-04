import json
import os
import numpy as np
import tensorflow as tf

def init():
    """
    Function called once when the deployment container is started. It loads the trained model from the Azure ML model directory.
    """
    
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(model_dir, "outputs", "best_overall_model.keras")
    model = tf.keras.models.load_model(model_path)

def run(raw_data):
    """
    Function called for each request to the deployment endpoint. It processes the input data, performs preprocessing, and returns the prediction results.
    """
    try:
        # 1. Parse input JSON data
        data = json.loads(raw_data)
        
        # Extract Record_Id, Okno, and Sygnal from the input data
        record_id = data.get('Record_Id', 'UNKNOWN')
        okno = data.get('Okno', 0)
        signal = np.array(data.get('Sygnal', []))
        
        # Check if the signal has the expected length of 1024 samples
        if len(signal) != 1024:
            return {
                "Record_Id": record_id,
                "Okno": okno,
                "error": f"Oczekiwano 1024 próbek, otrzymano {len(signal)}."
            }
        
        # 2. Preprocessing (Z-score)
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            signal = (signal - mean) / std
            
        # 3. Reshape the signal for model input
        input_tensor = signal.reshape(1, 1024, 1).astype(np.float32)
        
        # 4. Prediction
        prediction_prob = float(model.predict(input_tensor)[0][0])
        # Determine if the signal is normal or anomalous based on the prediction probability
        is_normal = True if prediction_prob > 0.5 else False
        is_anomaly_bool = not is_normal
        
      
        anomaly_probability = 1.0 - prediction_prob if is_normal else prediction_prob
        
        return {
            "Record_Id": record_id,
            "Okno": okno,
            "Prediction": "Anomaly" if is_anomaly_bool else "Normal",
            "Probability": round(anomaly_probability, 4),
            "Is_Anomaly": 1 if is_anomaly_bool else 0
        }
        
    except Exception as e:
        return {"error": str(e)}