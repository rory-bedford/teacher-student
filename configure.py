"""Point every experiment at your own data directory.

Each figure's ``experiment.toml`` holds absolute paths: the script and parameters file in
this repository, the folder its runs are written to, and the teacher data it reads. This
rewrites them all around one data directory of your choosing:

    uv run python configure.py /path/to/data              # W&B logging off
    uv run python configure.py /path/to/data --wandb my-project [--entity my-team]

Afterwards the teacher is written to ``<data>/teacher-activity`` and each figure's runs to
``<data>/<figure folder name>``. Only the path and W&B lines change; comments and every
other setting are kept. Run it again whenever you move the repository or the data.
"""

import argparse
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent
TEACHER = REPO / "fig00-teacher-activity"
TEACHER_OUTPUT = "teacher-activity"
#: The teacher files every figure symlinks as its input.
TEACHER_FILES = ("network_structure.npz", "spike_data.zarr")
LOG_FILE = "simulator_log.jsonl"


def set_value(text, key, value, section=None):
    """Replace ``key = ...`` (first occurrence, in ``[section]`` if given), keeping comments."""
    start = 0
    if section is not None:
        match = re.search(rf"^\[{re.escape(section)}\]\s*$", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"no [{section}] table")
        start = match.end()
    pattern = re.compile(
        rf"^(\s*{re.escape(key)}\s*=\s*)([^#\n]*?)(\s*(#.*)?)$", re.MULTILINE
    )
    match = pattern.search(text, start)
    if match is None:
        raise ValueError(f"no {key} entry")
    literal = (
        value if isinstance(value, str) and value in ("true", "false") else f'"{value}"'
    )
    return text[: match.start(2)] + literal + text[match.end(2) :]


def set_inputs(text, paths):
    """Rewrite the ``path`` of each ``[[data.inputs]]`` entry, in order."""
    matches = list(re.finditer(r'^(path\s*=\s*)"[^"]*"', text, re.MULTILINE))
    if len(matches) != len(paths):
        raise ValueError(
            f"expected {len(paths)} [[data.inputs]] paths, found {len(matches)}"
        )
    for match, path in reversed(list(zip(matches, paths))):
        text = text[: match.start()] + f'{match.group(1)}"{path}"' + text[match.end() :]
    return text


def configure(config, data_dir, wandb_project, entity):
    text = config.read_text()
    experiment = tomllib.loads(text)
    folder = config.parent
    is_teacher = folder == TEACHER
    output = data_dir / (TEACHER_OUTPUT if is_teacher else folder.name)

    text = set_value(text, "script", folder / Path(experiment["script"]).name)
    text = set_value(text, "output_dir", output)
    text = set_value(text, "parameters_file", folder / "parameters.toml")
    text = set_value(text, "log_file", data_dir / LOG_FILE)
    if not is_teacher:
        text = set_inputs(
            text,
            [data_dir / TEACHER_OUTPUT / "results" / name for name in TEACHER_FILES],
        )
    if "wandb" in experiment:
        text = set_value(text, "enabled", "true" if wandb_project else "false", "wandb")
        if wandb_project:
            text = set_value(text, "project", wandb_project, "wandb")
            text = set_value(text, "entity", entity or "", "wandb")
    tomllib.loads(text)  # still valid TOML
    config.write_text(text)
    return output


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_dir", type=Path, help="where teacher data and runs go")
    parser.add_argument(
        "--wandb", metavar="PROJECT", help="log to this W&B project (default: off)"
    )
    parser.add_argument("--entity", help="W&B entity (user or team)")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    for config in sorted(REPO.glob("fig*/experiment.toml")):
        output = configure(config, data_dir, args.wandb, args.entity)
        print(f"{config.relative_to(REPO)} -> {output}")
    logging = f"W&B project {args.wandb}" if args.wandb else "W&B off"
    print(f"\nData directory: {data_dir} ({logging})")


if __name__ == "__main__":
    main()
