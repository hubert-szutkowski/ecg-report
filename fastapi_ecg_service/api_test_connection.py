import requests
import numpy as np
import time

# Zdefiniuj kształt zgodny z oknem czasowym twojego wejścia EKG
# Zakładam standardowe wejście jednowymiarowe w batchu (1, L)
input_length = 1000 
dummy_signal = np.random.randn(1, input_length).tolist()

payload = {
    "signal": dummy_signal
    # Dodaj inne parametry, jeśli zdefiniowałeś je w schemacie Pydantic
}

url = "http://localhost:8000/predict"  # Uzupełnij o poprawny routing

start_time = time.time()
response = requests.post(url, json=payload)
latency = time.time() - start_time

if response.status_code == 200:
    print(f"Sukces. Czas inferencji: {latency:.4f} s")
    print("Predykcja:", response.json())
else:
    print(f"Błąd struktury żądania (Kod {response.status_code})")
    print("Szczegóły:", response.text)
    