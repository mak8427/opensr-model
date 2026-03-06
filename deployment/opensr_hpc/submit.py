from __future__ import annotations

from pathlib import Path
from typing import Any

from deployment.opensr_hpc.config import RuntimeConfig, runtime_config_to_dict
from deployment.opensr_hpc.manifests import new_run_id, write_yaml
from deployment.opensr_hpc.naming import patch_dir, resolve_run_dir
from deployment.opensr_hpc.patching import Patch
from deployment.opensr_hpc.slurm import SlurmJobSpec, submit_job
from deployment.opensr_hpc.staging import stage_cutout


def _patch_manifest(
    *,
    patch: Patch,
    run_id: str,
    start_date: str,
    end_date: str,
    config: RuntimeConfig,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "patch_id": patch.patch_id,
        "patch_index": patch.row_index * patch.column_count + patch.column_index,
        "latitude": patch.latitude,
        "longitude": patch.longitude,
        "edge_size": patch.edge_size,
        "start_date": start_date,
        "end_date": end_date,
        "paths": {
            "run_dir": "../..",
            "input_tif": "inputs/lr.tif",
            "output_dir": "outputs",
            "metadata_dir": "metadata",
        },
        "config": runtime_config_to_dict(config),
    }


def submit_patch_run(
    *,
    config: RuntimeConfig,
    patch: Patch,
    start_date: str,
    end_date: str,
    script_path: Path,
    dry_run: bool = False,
) -> tuple[str, Path, dict[str, str]]:
    run_id = new_run_id(config.project_name)
    run_dir = resolve_run_dir(config.output_root, run_id)
    logs_dir = run_dir / "logs"
    patch_root = patch_dir(run_dir, patch.patch_id)
    input_tif = patch_root / "inputs" / "lr.tif"

    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "resolved_config.yaml", runtime_config_to_dict(config))

    if not dry_run:
        stage_cutout(
            latitude=patch.latitude,
            longitude=patch.longitude,
            start_date=start_date,
            end_date=end_date,
            config=config.staging,
            output_path=input_tif,
        )
    else:
        input_tif.parent.mkdir(parents=True, exist_ok=True)

    manifest = _patch_manifest(
        patch=patch,
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    write_yaml(
        run_dir / "run_manifest.yaml",
        {
            "run_id": run_id,
            "mode": "patch",
            "patch_count": 1,
            "tasks": [{"patch_id": patch.patch_id, "manifest": f"patches/{patch.patch_id}/manifest.yaml"}],
        },
    )
    manifest_path = patch_root / "manifest.yaml"
    write_yaml(manifest_path, manifest)

    spec = SlurmJobSpec(
        job_name=f"opensr_{patch.patch_id}",
        script_path=script_path,
        manifest_path=manifest_path,
        output_path=logs_dir / f"slurm_{patch.patch_id}.out",
        error_path=logs_dir / f"slurm_{patch.patch_id}.err",
        slurm=config.slurm,
        environment=config.environment,
    )
    submission = submit_job(spec, run_dir / "submission", dry_run=dry_run)
    return run_id, run_dir, submission


def submit_grid_run(
    *,
    config: RuntimeConfig,
    patches: list[Patch],
    start_date: str,
    end_date: str,
    script_path: Path,
    dry_run: bool = False,
) -> tuple[str, Path, dict[str, str]]:
    run_id = new_run_id(config.project_name)
    run_dir = resolve_run_dir(config.output_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "resolved_config.yaml", runtime_config_to_dict(config))

    tasks: list[dict[str, object]] = []
    for patch_index, patch in enumerate(patches):
        patch_root = patch_dir(run_dir, patch.patch_id)
        input_tif = patch_root / "inputs" / "lr.tif"
        if not dry_run:
            stage_cutout(
                latitude=patch.latitude,
                longitude=patch.longitude,
                start_date=start_date,
                end_date=end_date,
                config=config.staging,
                output_path=input_tif,
            )
        else:
            input_tif.parent.mkdir(parents=True, exist_ok=True)

        manifest = _patch_manifest(
            patch=patch,
            run_id=run_id,
            start_date=start_date,
            end_date=end_date,
            config=config,
        )
        manifest["patch_index"] = patch_index
        write_yaml(patch_root / "manifest.yaml", manifest)
        tasks.append(manifest)

    run_manifest = {
        "run_id": run_id,
        "mode": "grid",
        "patch_count": len(tasks),
        "start_date": start_date,
        "end_date": end_date,
        "tasks": [
            {"patch_id": str(task["patch_id"]), "manifest": f"patches/{task['patch_id']}/manifest.yaml"}
            for task in tasks
        ],
    }
    write_yaml(run_dir / "run_manifest.yaml", run_manifest)

    spec = SlurmJobSpec(
        job_name=f"opensr_{run_id}",
        script_path=script_path,
        manifest_path=run_dir / "run_manifest.yaml",
        output_path=logs_dir / "slurm_%A_%a.out",
        error_path=logs_dir / "slurm_%A_%a.err",
        slurm=config.slurm,
        environment=config.environment,
        array=f"0-{len(tasks) - 1}" if tasks else None,
    )
    submission = submit_job(spec, run_dir / "submission", dry_run=dry_run)
    return run_id, run_dir, submission
