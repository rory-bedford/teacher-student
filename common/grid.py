"""Grid-search helper: skip runs that already finished.

Lets a grid be relaunched after a crash or a pilot without redoing completed runs.
A run directory that exists but never finished is an error, because the run framework
would stop to ask whether to overwrite it, which an unattended grid cannot answer.
"""

from pathlib import Path

import toml

from common.training import FINAL_STATE


def skip_completed(experiment_config_path, config_generator, unfinished="error"):
    """Wrap ``config_generator`` so it only yields runs that have not finished.

    Args:
        unfinished: What to do with run directories that exist but never finished —
            ``"error"`` (stop, for launching a grid) or ``"skip"`` (warn and leave them
            out, e.g. a run still in progress when preparing a SLURM array).
    """
    grid_dir = Path(toml.load(experiment_config_path)["output_dir"])

    def generator(base_params):
        unfinished_dirs = []
        for params, description in config_generator(base_params):
            run_dir = grid_dir / description
            if (run_dir / FINAL_STATE).exists():
                print(f"Skipping completed run: {description}")
                continue
            if run_dir.exists():
                unfinished_dirs.append(str(run_dir))
                continue
            yield params, description
        if unfinished_dirs and unfinished == "skip":
            print(
                "Skipping run directories that exist but have not finished "
                "(still running, or crashed):\n  " + "\n  ".join(unfinished_dirs)
            )
        elif unfinished_dirs:
            raise SystemExit(
                "These run directories exist but never finished; move or delete them "
                "and relaunch:\n  " + "\n  ".join(unfinished_dirs)
            )

    return generator
