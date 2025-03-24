import wandb
from typing import Dict, Optional

def init_wandb(project_name: str, config: Dict, run_name: Optional[str] = None, entity:Optional[str]=None):
    """Initializes a wandb run."""
    if wandb.run is not None: #Close any existing run
        wandb.finish()
    run = wandb.init(project=project_name, config=config, name=run_name, entity=entity)
    return run

def log_wandb(metrics: Dict):
    """Logs metrics to wandb."""
    if wandb.run is not None:
        wandb.log(metrics)

def save_wandb_artifact(path: str, artifact_name: str, artifact_type: str):
     """Saves an artifact to wandb."""
     if wandb.run is not None:
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(path)
        wandb.log_artifact(artifact)