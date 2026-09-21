# coding: utf-8
"""Where things live, and how the analysis stages write their own tree.

Every path comes from an environment variable, so a different machine needs no
code edit and a run records what it actually used.  The defaults are relative to
the repository root (`env.sh` sets the same values explicitly):

    DATA_ROOT    <repo>/datasets
    MODEL_ROOT   <repo>/models/Qwen3            Qwen3-1.7B / Qwen3-4B / Qwen3-8B
    TRAIN_ROOT   <repo>/outputs/checkpoints     one directory per trained run
    EVAL_ROOT    <repo>/outputs/eval            responses / events / summary / e1_metrics.csv
    PAPER_ROOT   <repo>/outputs/paper           everything the analysis stages write
    SAE_ROOT     <repo>/models/SAE-Res-Qwen3-8B-Base-W64K-L0_100
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _env_path(name: str, default: Path, *aliases: str) -> Path:
    for key in (name, *aliases):
        value = os.environ.get(key, "").strip()
        if value:
            return Path(value)
    return default


@dataclass(frozen=True)
class Layout:
    """Absolute inputs and outputs for one analysis run."""

    repo_root: Path
    data_root: Path
    e1_root: Path            # trained checkpoints, one directory per run name
    eval_root: Path          # responses / events / summary / e1_metrics.csv
    paper_root: Path         # everything the analysis stages write
    model_root: Path         # base Qwen3 checkpoints (the untuned references)
    sae_root: Path

    @classmethod
    def from_env(cls) -> "Layout":
        outputs = REPO_ROOT / "outputs"
        return cls(
            repo_root=REPO_ROOT,
            data_root=_env_path("DATA_ROOT", REPO_ROOT / "datasets"),
            e1_root=_env_path("TRAIN_ROOT", outputs / "checkpoints", "E1_ROOT"),
            eval_root=_env_path("EVAL_ROOT", outputs / "eval"),
            paper_root=_env_path("PAPER_ROOT", outputs / "paper"),
            model_root=_env_path("MODEL_ROOT", REPO_ROOT / "models" / "Qwen3"),
            sae_root=_env_path(
                "SAE_ROOT",
                REPO_ROOT / "models" / "SAE-Res-Qwen3-8B-Base-W64K-L0_100"),
        )

    # ---------------------------------------------------------------- inputs

    @property
    def metrics_csv(self) -> Path:
        return self.eval_root / "e1_metrics.csv"

    @property
    def gold(self) -> Path:
        return self.data_root / "cleanv2" / "eval_supportclean_keep8.jsonl"

    def responses(self, tag: str) -> Path:
        return self.eval_root / "responses" / f"{tag}_nothink_n8.jsonl"

    def events(self, tag: str) -> Path:
        return self.eval_root / "events" / f"{tag}_event_rows.jsonl"

    def summary(self, tag: str) -> Path:
        return self.eval_root / "summary" / f"{tag}_summary.json"

    def checkpoint(self, tag: str) -> Path:
        """The final checkpoint directory of one trained run.

        The step number is discovered rather than assumed: `checkpoint-417` is
        what this round happened to produce, not a naming rule.
        """
        run_dir = self.e1_root / tag
        steps = sorted(
            (int(child.name.split("-")[1]), child)
            for child in run_dir.glob("checkpoint-*")
            if child.is_dir() and child.name.split("-")[-1].isdigit()
            and (child / "config.json").is_file())
        if steps:
            return steps[-1][1]
        if (run_dir / "config.json").is_file():
            return run_dir
        raise FileNotFoundError(
            f"no loadable checkpoint under {run_dir}; the metrics row's "
            "model_path column is the fallback source for this tag")

    def base_model(self, size: str) -> Path:
        return self.model_root / f"Qwen3-{size}"

    # --------------------------------------------------------------- outputs

    @property
    def r0(self) -> Path:
        return self.paper_root / "R0"

    @property
    def r1(self) -> Path:
        return self.paper_root / "R1"

    @property
    def r2(self) -> Path:
        return self.paper_root / "R2"

    @property
    def x1(self) -> Path:
        return self.paper_root / "X1"

    @property
    def tables(self) -> Path:
        return self.paper_root / "tables"

    @property
    def figures(self) -> Path:
        return self.paper_root / "figures"

    @property
    def appendix(self) -> Path:
        return self.paper_root / "appendix"

    @property
    def compact(self) -> Path:
        return self.paper_root / "compact"

    def ensure(self) -> None:
        for path in (self.paper_root, self.r0, self.r1, self.r2, self.x1,
                     self.tables, self.figures, self.appendix, self.compact):
            path.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, str]:
        return {
            "repo_root": str(self.repo_root),
            "data_root": str(self.data_root),
            "e1_root": str(self.e1_root),
            "eval_root": str(self.eval_root),
            "paper_root": str(self.paper_root),
            "model_root": str(self.model_root),
            "sae_root": str(self.sae_root),
        }


def model_path_for(layout: Layout, tag: str, *, metrics_row: dict | None = None) -> Path:
    """Checkpoint of one evaluation tag: the metrics row first, the tree as fallback.

    `e1_metrics.csv` records the exact directory each row was scored from, so
    it is the authoritative answer whenever the row exists.
    """
    if metrics_row:
        candidate = str(metrics_row.get("model_path") or "").strip()
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    from . import registry
    if registry.BY_TAG.get(tag) and registry.BY_TAG[tag].condition == "notrain":
        return layout.base_model(registry.BY_TAG[tag].size)
    return layout.checkpoint(tag)
