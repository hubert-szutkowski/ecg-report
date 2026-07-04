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
        
        # Surowy wynik z modelu to prawdopodobieństwo klasy 0 (Normal)
        normal_prob = float(model.predict(input_tensor)[0][0])

        # Obliczamy prawdopodobieństwo anomalii (klasy 1)
        anomaly_prob = 1.0 - normal_prob

        # Próg klasyfikacji ustawiony na szansę wystąpienia anomalii
        is_anomaly = 1 if anomaly_prob > 0.5 else 0

        return {
            "Record_Id": record_id,
            "Okno": okno,
            "Prediction": "Anomaly" if is_anomaly == 1 else "Normal",
            "Probability": round(anomaly_prob, 4), # Wysyłamy do Power BI szansę na anomalię
            "Is_Anomaly": is_anomaly
        }
        
    except Exception as e:
        return {"error": str(e)}