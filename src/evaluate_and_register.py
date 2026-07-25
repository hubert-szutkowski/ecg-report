import argparse
import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import sys

def main():
    parser = argparse.ArgumentParser(description="Evaluate Challenger model against the registered Champion")
    parser.add_argument("--experiment-name", type=str, default="ecg-anomaly-difference", help="Azure ML experiment name")
    parser.add_argument("--model-name", type=str, default="ecg_multiclass_model", help="Registered model name")
    parser.add_argument("--metric-name", type=str, default="cv_mean_val_auc", help="The exact name of the manually logged metric to compare")
    parser.add_argument("--direction", type=str, choices=["maximize", "minimize"], default="maximize", help="Whether to maximize (e.g., AUC) or minimize (e.g., loss) the metric")
    args = parser.parse_args()

    # Init Azure ML and MLflow
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    workspace = ml_client.workspaces.get(name=ml_client.workspace_name)
    mlflow.set_tracking_uri(workspace.mlflow_tracking_uri)

    # Load latest challenger run
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if not experiment:
        print(f"CRITICAL: Experiment '{args.experiment_name}' not found.")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if runs.empty:
        print("CRITICAL: No runs found for this experiment.")
        return

    latest_run = runs.iloc[0]
    challenger_run_id = latest_run.run_id
    
    # Read the target metric
    metric_column_name = f"metrics.{args.metric_name}"
    
    if metric_column_name not in latest_run:
        print(f"CRITICAL: Metric '{args.metric_name}' was not found in the latest run. Please check your training script.")
        return
        
    challenger_score = float(latest_run[metric_column_name])
    print(f"Challenger Run ID: {challenger_run_id}")
    print(f"Challenger score ({args.metric_name}): {challenger_score:.4f}")

    # Load current champion score
    best_registered_score = -float('inf') if args.direction == "maximize" else float('inf')
    
    try:
        champion_model = ml_client.models.get(name=args.model_name, label="latest")
        best_registered_score = float(champion_model.tags.get(args.metric_name, best_registered_score))
        print(f"Current Champion model version: {champion_model.version}")
        print(f"Current Champion score ({args.metric_name}): {best_registered_score:.4f}")
    except Exception:
        print("INFO: No registered Champion model found. This will be the first deployment.")

    # Compare challenger vs champion
    is_better = False
    if args.direction == "maximize":
        is_better = challenger_score > best_registered_score
    else:
        is_better = challenger_score < best_registered_score

    if is_better:
        print(f"SUCCESS: Challenger outperformed the Champion ({challenger_score:.4f} vs {best_registered_score:.4f})!")
        print("Proceeding with model registration...")
        
        model_uri = f"azureml://jobs/{challenger_run_id}/outputs/artifacts/paths/outputs/ecg_multiclass_model.keras"
        
        new_champion = Model(
            path=model_uri,
            name=args.model_name,
            description="InceptionTime ECG Multiclass Classifier",
            type=AssetTypes.CUSTOM_MODEL,
            tags={
                args.metric_name: str(challenger_score),
                "run_id": challenger_run_id,
                "framework": "keras_tensorflow"
            }
        )
        
        ml_client.models.create_or_update(new_champion)
        print(f"SUCCESS: Registered new Champion version with {args.metric_name}: {challenger_score:.4f}")
        sys.exit(0)  # Success
    else:
        print(f"REJECTED: Challenger score ({challenger_score:.4f}) did not beat Champion score ({best_registered_score:.4f}).")
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()