from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
)
from azure.identity import InteractiveBrowserCredential
import json
import os
import sys


print("Logging to Azure...")
credential = InteractiveBrowserCredential()

try:
    
    ml_client = MLClient.from_config(credential=credential)
    print("Success! Logged in to Azure ML using the config file.")
except Exception as e:
    print(f"❌ ERROR: Unable to initialize MLClient from config.json: {e}")
    print("Make sure the config.json file downloaded from Azure is in the project root directory.")
    sys.exit(1)


print("Loading SQL configuration from the common file...")
with open("config.json", "r") as f:
    full_config = json.load(f)

sql_env_vars = {
    key: value for key, value in full_config.items() 
    if key.startswith("SQL_")
}
print(f"SQL configuration loaded successfully (found keys: {len(sql_env_vars)}).")


ENDPOINT_NAME = "ecg-holter-api-hs"  
MODEL_NAME = "ecg-anomaly-detector"
MODEL_VERSION = "2" 

print(f"Creating/Updating Endpoint: {ENDPOINT_NAME}...")
endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    description="API for the  Holter reader system with automatic workspace configuration",
    auth_mode="key",
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Endpoint ready!")


print("Building machine and environment in the cloud (approx. 10-15 minutes)...")

env = Environment(
    name="ecg-inference-sql-env",
    version="2",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    conda_file="conda.yml"
)

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=ENDPOINT_NAME,
    model=f"azureml:{MODEL_NAME}:{MODEL_VERSION}",
    environment=env,
    code_configuration=CodeConfiguration(
        code="./src",               
        scoring_script="score.py"   
    ),
   
    environment_variables=sql_env_vars,
    instance_type="Standard_F2s_v2", 
    instance_count=1,
)

ml_client.online_deployments.begin_create_or_update(deployment).result()


print("Routing 100% traffic to the 'blue' deployment...")
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print(f"\n✅ SUCCESS! The production server has been successfully deployed.")