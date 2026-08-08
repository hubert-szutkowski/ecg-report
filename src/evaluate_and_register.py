import argparse
import json
import sys
import tempfile
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import InteractiveBrowserCredential


def build_parser():
    parser = argparse.ArgumentParser(description="Register a new Champion model based on the evaluation job's promotion decision")
    parser.add_argument("--eval-job-name", type=str, required=True, help="Azure ML evaluation job name whose promotion_decision.json to read")
    parser.add_argument("--train-job-name", type=str, required=True, help="Azure ML training job name to register as the new Champion, if promoted")
    parser.add_argument("--model-name", type=str, default="ecg_multiclass_model", help="Registered model name")
    return parser


def load_promotion_decision(ml_client: MLClient, eval_job_name: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        ml_client.jobs.download(name=eval_job_name, download_path=tmp_dir)
        try:
            decision_path = next(Path(tmp_dir).rglob("promotion_decision.json"))
        except StopIteration:
            raise RuntimeError(f"promotion_decision.json not found in outputs of job '{eval_job_name}'")
        return json.loads(decision_path.read_text())


def main():
    args = build_parser().parse_args()

    print("Logging to Azure...")
    ml_client = MLClient.from_config(credential=InteractiveBrowserCredential())

    decision = load_promotion_decision(ml_client, args.eval_job_name)
    print(decision["reason"])

    if not decision["promote"]:
        print("REJECTED: challenger did not beat the champion.")
        sys.exit(1)

    print("SUCCESS: promoting challenger to Champion...")

    model_uri = f"azureml://jobs/{args.train_job_name}/outputs/artifacts/paths/outputs/ecg_multiclass_model.keras"

    new_champion = Model(
        path=model_uri,
        name=args.model_name,
        description="InceptionTime ECG Multiclass Classifier",
        type=AssetTypes.CUSTOM_MODEL,
        tags={
            **{k: str(v) for k, v in decision["challenger_metrics"].items()},
            "run_id": args.train_job_name,
            "framework": "keras_tensorflow",
        },
    )

    ml_client.models.create_or_update(new_champion)
    print(f"SUCCESS: Registered new Champion model version for run {args.train_job_name}")
    sys.exit(0)


if __name__ == "__main__":
    main()
