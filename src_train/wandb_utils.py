import tempfile
from pathlib import Path

import wandb


def save_and_log_model(lit_module, variant: str = "last") -> None:
    """Save decoder weights to the current wandb run's files directory."""
    wandb_run_dir = Path(wandb.run.dir)
    save_path = wandb_run_dir / "individual_components"
    save_path.mkdir(exist_ok=True)

    lit_module.save_individual_components(save_path, variant=variant)

    for component in lit_module._SAVE_COMPONENTS:
        filename = f"{component}_{variant}.pth"
        full_path = save_path / filename
        if full_path.exists():
            print(f"Logging {filename} to wandb.")
            wandb.save(str(full_path), base_path=str(wandb_run_dir))


def _download_component(run, remote_name: str, local_path: Path) -> bool:
    """Download one component file from a wandb run. Returns False on 404."""
    try:
        run.file(f"individual_components/{remote_name}").download(
            root=str(local_path.parent), replace=True
        )
        downloaded = local_path.parent / "individual_components" / remote_name
        downloaded.rename(local_path)
        return True
    except wandb.errors.CommError as e:
        if "404" in str(e):
            return False
        raise


def load_pretrained_model(
    wandb_run_id: str,
    lit_module,
    variant: str = "best",
) -> None:
    """Download decoder weights from a wandb run and load them into lit_module."""
    api = wandb.Api()
    run = api.run(f"touchgpt/{wandb_run_id}")

    print(f"Downloading components from touchgpt/{wandb_run_id} (variant={variant})")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for component in lit_module._SAVE_COMPONENTS:
            remote_name = f"{component}_{variant}.pth"
            local_path = tmp_path / f"{component}.pth"
            print(f"  Downloading {remote_name}")
            if not _download_component(run, remote_name, local_path):
                print(f"  Warning: {remote_name} not found in run, skipping.")

        lit_module.load_individual_components(tmp_path)
