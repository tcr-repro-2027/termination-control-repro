# From Hallucinated Targets to Runaway Repetition: Termination Control in Long Structured Generation

Anonymous code release for the ICLR 2027 submission. It provides supervision
builders, the 20-run training matrix, generation and scoring code, and the
mechanism analyses, tables and figures. Training and evaluation require the
separately released data and model snapshots. Rebuilding some supervision
conditions requires optional upstream inputs; input-edit controls are frozen.

* **Self-contained.** One Python package (`tcr/`) plus numbered entry scripts
  (`experiments/`). Nothing is imported from outside this directory.
* **Tests included.** `bash run_tests.sh` runs the CPU test suites. Most checks
  require no model weights or data; tokenizer and process-group checks skip
  when their prerequisites are unavailable.
* **Reference outputs included.** `reference_results/` holds the CSV/JSON files behind
  every number in the paper, so each step can be compared with what we obtained.
* **Data previews included.** `datasets/` contains the first 10 records of each
  of the 28 data files for browsing their formats. Experiments require the full
  [dataset](https://huggingface.co/datasets/tcr-repro-2027/termination-control-data).

---

## 1. Layout

```
├── env.sh                      paths and runtime variables (source it first)
├── requirements.txt            CPU stages and tests
├── requirements-gpu.txt        training, generation and hooked forward passes
├── run_tests.sh
├── datasets/                   format previews: 10 records per file; not experiment inputs
├── tcr/                        library
│   ├── prompt_template.py      the single prompt used for training AND evaluation
│   ├── data/                   supervision construction
│   │   ├── stage_build/        corpus cleaning stages (cleaned supervision = `cleanv2`)
│   │   ├── raw_arms/           raw supervision (`keep4`) and the two violation-subtype arms
│   │   └── controlled/         out-of-candidate block replacement (OBR), generic label noise
│   ├── evaluation/             vLLM generation, per-response scoring, metrics table
│   ├── events/                 block parser, reuse events, continuous pattern repetition,
│   │                           two-stage competing-risk process and its standardization
│   ├── token_orbit/            stable-token-orbit detector
│   ├── extraction/             triple / entity-pair F1
│   ├── boundary/               close-vs-continue readouts on a fixed prefix (stop margin,
│   │                           reachability, next-event sampling, forced close -> EOS)
│   ├── steering/               contrast direction, single-pulse hook, SAE reading
│   ├── motif/                  artificial complete-motif probes and gain
│   ├── support_probe/          input-support probe (appendix)
│   └── paper/                  model registry, paths, paired bootstrap, figure style
├── experiments/
│   ├── 1_data/                 build_data.sh + builders
│   ├── 2_train/                run_train_all.sh, train_single.sh, DeepSpeed config
│   ├── 3_evaluate/             run_eval_all.sh + preflight / collect
│   ├── 4_analysis/             run_analysis.sh + one script per analysis
│   └── 5_support_probe/        run_support_probe.sh + preflight / collect / diagnose
├── tests/                      one sub-directory per sub-package
└── reference_results/          our outputs of steps 3-5 (small files only)
```

## 2. Paper -> code

| Paper | Command | Output (under `$PAPER_ROOT` unless noted) | Our file in `reference_results/` |
|---|---|---|---|
| Sec. 3.2, App. A: raw / cleaned supervision, OBR 5-24.3 %, controls | `bash experiments/1_data/build_data.sh all` | `$DATA_ROOT/{raw,controlled}/` | - (data archive) |
| App. I: corpus statistics, replacement dose and length | `build_data.sh stats` | `$DATA_ROOT/dataset_statistics/`, `controlled/obr_dose_data_table.csv` | - |
| Sec. 4.1, App. I: training configuration | `bash experiments/2_train/run_train_all.sh` | `$TRAIN_ROOT/<run>/checkpoint-417` | - |
| Sec. 4.2-4.3, Tab. 1, App. J-K (natural contrasts, OBR, controls) | `bash experiments/3_evaluate/run_eval_all.sh` then `run_analysis.sh assets` | `$EVAL_ROOT/e1_metrics.csv`; `tables/T1_*`, `tables/T2_*`, `tables/N1_*`, `tables/O1_*`, `appendix/A_controls*` | `input/e1_metrics.csv`, `tables/`, `appendix/A_*` |
| Sec. 3.3 / 4.4, Fig. 2, App. B and M (reuse dynamics) | `run_analysis.sh r0` | `R0/<pair>/p0d_episode_hazard/*.csv`, `R0/R0_pair_metrics.csv` | `R0/` |
| Sec. 3.4, App. C (natural prefix pools) | `run_analysis.sh pool` | `R1/prefix_pool_<scale>.jsonl` and its report | `compact/R1_pool.txt` |
| Sec. 4.5, Fig. 3, App. D and O (termination on shared prefixes) | `run_analysis.sh r1` | `R1/R1_paired_effects_<scale>.csv`, `R1/R1_close_execution_<scale>.csv` | `R1/` |
| Sec. 3.5 / 4.6, Fig. 4, Tab. 2, App. E and P (direction, single pulse, full continuations) | `run_analysis.sh r2` | `R2/R2_direction_dev_*`, `R2_short_effects_*`, `R2_projection_test_*`, `R2_quality_tradeoff_4B.csv`, `R2_long_*` | `R2/` |
| App. G (8B sparse feature reading) | `run_analysis.sh r2` (8B branch) | `R2/R2_sae_features_8B.csv`, `appendix/D_sae_features.*` | `R2/`, `appendix/D_*` |
| App. F and N (artificial complete-motif gain) | `run_analysis.sh x1` | `X1/X1_probe_scores.csv`, `X1_paired_effects.csv`, `appendix/B_motif_gain*` | `X1/`, `appendix/B_*` |
| App. L (input-support probe) | `bash experiments/5_support_probe/run_support_probe.sh` | `$PROBE_ROOT/e2_metrics.csv`; `appendix/C_e2_limits.csv` | `input/e2_metrics.csv`, `appendix/C_*` |
| Fig. 1-4 and the two process-curve figures of App. M | `run_analysis.sh figures` | `figures_paper/F1_overview` ... `F4_intervention`, `A1_process_curves`, `A2_process_curves` (`.pdf`, `.png`) | `figures/*.png` |

`compact/*.txt` are one-screen plain-text summaries of each stage; they are the quickest
way to compare a rerun with ours.

The six figures of the paper are drawn by `experiments/4_analysis/make_figures.py` from
result CSVs only; `reference_results/figures/` holds its output on our results. (`assets`
additionally writes diagnostic plots of the same quantities to `figures/`.)

### Names used in the code

The identifiers predate the paper's terminology. They are kept because they are stamped
into file names and result columns.

| Paper | Code |
|---|---|
| cleaned supervision | `cleanv2` (file `train_supportclean_keep8.jsonl`) |
| raw supervision | `keep4` |
| raw, candidate-only subtype / out-of-text subtype | `keep4_a` / `keep4_ae` |
| OBR 24.3 % / 15 % / 10 % / 5 % | `obr` / `obr_p15` / `obr_p10` / `obr_p5` |
| candidate / text / both input edits, benign input edit | `isc_a` / `isc_e` / `isc_ae`, `benign_input` |
| generic label noise | `generic_noise` |
| untuned model | `notrain` |
| continuous pattern repetition (capture) | `semantic_capture`, `motif_capture_triple` |
| stable token orbit | `stable_orbit`, `legacy_orbit` |
| context-limit ending | `hit_max` |
| triple F1 / entity-pair F1 | `strict_*` / `relaxed_*` |
| reuse-dynamics analysis, its process / set-completion analysers | `R0`, `P0d` / `P0c` |
| exposure / persistence component | `exposure_component` / `propensity_component` |
| shared-prefix termination | `R1`; roles `natural_stop`, `early_remaining`, `pre_first_reuse` |
| contrast direction and interventions | `R2`; conditions `baseline`, `direction`, `random_*`, `close_bias` |
| complete-motif gain | `X1` |
| input-support probe; candidate / text axis | `E2` (`e2_*`); `aci_*` / `eci_*` |
| natural-generation evaluation | `E1` (`e1_*`) |

A model tag is the training run name, e.g. `qwen3-4b-keep4-s123`. Two tags carry no seed
suffix (`qwen3-1.7b-cleanv2`, `qwen3-8b-cleanv2`); their training seed is 42. The full
registry, with the role of each model and every paired contrast, is
`tcr/paper/registry.py`.

## 3. Setup

```bash
conda create -n tcr python=3.11 -y && conda activate tcr
pip install -r requirements.txt          # CPU stages + tests
pip install -r requirements-gpu.txt      # training / generation / hooks
source env.sh
bash run_tests.sh
```

**Models.** Put Hugging Face snapshots of `Qwen3-1.7B`, `Qwen3-4B` and `Qwen3-8B` (the
instruction models) under `$MODEL_ROOT` (default `models/Qwen3/`). The 8B feature reading
additionally needs the released sparse autoencoder
`SAE-Res-Qwen3-8B-Base-W64K-L0_100` at `$SAE_ROOT`; without it that one stage is reported
as not measured and everything else runs.

**Data previews.** The bundled `datasets/` directory contains exactly the first
10 records from each of the 28 full data files, copied in their original order.
These files are only for inspecting the formats. They are not representative
samples or sufficient inputs for training, evaluation, or paper reproduction.
The 10-row OBR manifest need not refer to the preview records in the other files.
See [datasets/README.md](datasets/README.md) for the preview layout and schemas.

**Full data.** Use the anonymous dataset repository:

> [tcr-repro-2027/termination-control-data](https://huggingface.co/datasets/tcr-repro-2027/termination-control-data)

From the code repository root, download into a separate directory and configure
the experiment drivers to read it. This keeps the small previews intact:

```bash
HF_HUB_OFFLINE=0 hf download tcr-repro-2027/termination-control-data \
  --repo-type dataset --local-dir full_data/datasets --exclude ".gitattributes"
export DATA_ROOT="$PWD/full_data/datasets"
source env.sh
```

The experiment drivers reject inputs inside a directory marked
`PREVIEW_ONLY.json`. Point `DATA_ROOT` at the complete download, not the
bundled preview directory. A different full-data location is also supported;
step 1 expects its last directory component to be named `datasets`.

The full input layout is:

```text
full_data/datasets/             DATA_ROOT
├── README.md                  full dataset card and applicable terms
├── cleanv2/
│   ├── train_supportclean_keep8.jsonl        8,854 records
│   ├── eval_supportclean_keep8.jsonl         1,106 records
│   └── swift_train_supportclean_keep8.jsonl
├── raw/
│   ├── train_keep4{,_a,_ae}.jsonl
│   └── swift_train_keep4{,_a,_ae}.jsonl
└── controlled/
    ├── train_{obr,obr_p5,obr_p10,obr_p15,isc_a,isc_e,isc_ae,benign_input,generic_noise}.jsonl
    ├── swift_train_<same nine suffixes>.jsonl
    └── obr_pair_manifest.jsonl             142,162 entries
```

`cleanv2/`, `raw/`, and `controlled/` are direct children of `DATA_ROOT`.
There is no extra `stages/` or `datasets/` level around the cleaned data.
Before running an experiment, verify that the full cleaned training and
evaluation files have 8,854 and 1,106 records rather than 10. Record the dataset
commit SHA used for a run; add `--revision` with that SHA to the download
command when reproducing the same version. If the full release includes
`SHA256SUMS`, verify it from `DATA_ROOT` with `sha256sum -c SHA256SUMS`.

Record form is one JSON object per line: `text` (source document), `entities_str`
(candidate entity list), `output` (list of `{source, target, relation, description}`);
evaluation rows add `key` and `source`. SWIFT form is
`{"messages": [user prompt, assistant JSON list]}` rendered with `tcr/prompt_template.py`.

**Hardware.** Training and the sharded analyses assume one node with 8 GPUs. Generation
and the hooked forward passes load one model per GPU (no tensor parallelism), so a single
GPU must hold an 8B model with a 32,768-token context. Set `GPUS=0,1,2,3` etc. to use
fewer devices for steps 3-5; training keeps a global batch of 64 only with 8 GPUs (see the
warning in `train_single.sh`).

## 4. Reproduction, step by step

Every driver has a `--check` / `check` mode that lists missing inputs and computes
nothing. All stages are resumable: finished work is detected from sentinel files on disk
and skipped, so after an interruption the same command is simply launched again.

### Step 1 - supervision conditions (CPU, optional)

The data release contains every training file used by the run matrix.
The following commands rebuild the deterministic conditions when the required
upstream files and tokenizer are available:

```bash
bash experiments/1_data/build_data.sh raw       # raw supervision + subtype arms (+ verification)
bash experiments/1_data/build_data.sh obr       # OBR 24.3 %: allocation, matching, invariants
bash experiments/1_data/build_data.sh doses     # nested 15 / 10 / 5 % subsets
bash experiments/1_data/build_data.sh noise     # generic label noise
bash experiments/1_data/build_data.sh formats   # record form -> SWIFT chat form
bash experiments/1_data/build_data.sh stats     # corpus statistics, replacement-dose table
```

`raw` and `obr` need optional upstream files beyond the 28 core files
(`stages/filter`, `stages/clean`, row maps, `build_reports/filter_clean_problem4_diff.jsonl`).
`noise` needs `controlled/edit_anchor_manifest.jsonl`, which is not in the core
release. `doses`, `formats` and `stats` use the core release and a tokenizer.
The lower OBR doses are
prefixes of one stratified ordering of the 24.3 % pairing (5 % within 10 % within 15 %);
`build_obr5.py` re-derives the 10 % and 15 % prefixes and refuses to continue unless they
match the released files block for block. The four input-edit controls are released as
frozen files only: their edits were drafted with an LLM under validation rules and are not
a deterministic function of the corpus.

### Step 2 - training (8 GPUs)

```bash
bash experiments/2_train/run_train_all.sh --check
nohup bash experiments/2_train/run_train_all.sh > train.log 2>&1 &
# a subset:  ONLY=qwen3-4b-cleanv2-s42,qwen3-4b-keep4-s42 bash experiments/2_train/run_train_all.sh
```

Full-parameter SFT with ms-swift 4.4.2: bf16, 3 epochs, learning rate 1e-5, cosine
schedule, warmup ratio 0.03, weight decay 0.1, Adam beta2 0.95, gradient clipping 1,
8 GPUs x micro-batch 1 x 8 accumulation steps = 64, unpacked sequences up to 32,768 tokens
(over-length examples deleted), DeepSpeed ZeRO-3 (CPU optimizer offload at 8B only),
seed = data seed. Every dataset yields 417 optimizer steps; the final checkpoint is the
one evaluated. `run_train_all.sh` holds the run matrix (20 rows) and is also the model
registry that steps 3 and 5 parse, so what gets evaluated is exactly what was trained.

### Step 3 - natural generation and scoring (GPU + CPU)

```bash
bash experiments/3_evaluate/run_eval_all.sh --check
nohup bash experiments/3_evaluate/run_eval_all.sh > eval.log 2>&1 &
# smoke test (about ten minutes):
LIMIT=8 ONLY=qwen3-4b-notrain EVAL_ROOT=outputs/eval_smoke bash experiments/3_evaluate/run_eval_all.sh
```

20 trained models + 3 untuned references, 1,106 documents x 8 sampling seeds = 8,848
responses per model, vLLM, non-thinking mode, temperature 0.7, top-p 0.8, top-k 20,
presence penalty 1.5, repetition penalty 1.0, generation to EOS or the 32,768-token total
context. The protocol is code (`tcr/evaluation/protocol.py`) and is stamped, together with
the sha256 of the evaluation file and of the rendered prompt, into every output; the final
`e1_collect.py` pass refuses to call rows comparable if any of them differ.

Outputs under `$EVAL_ROOT`: `responses/<tag>_nothink_n8.jsonl` (raw text),
`events/<tag>_event_rows.jsonl` (one row per response: parsed blocks, reuse events,
repetition, orbit, extraction counts, support counts), `summary/<tag>_summary.json`,
and `e1_metrics.csv` (one row per model). Roughly 0.3-0.6 GB per model.

### Step 4 - mechanism analyses, tables and figures

```bash
bash experiments/4_analysis/run_analysis.sh check
bash experiments/4_analysis/run_analysis.sh r0          # CPU
bash experiments/4_analysis/run_analysis.sh pool        # CPU + tokenizer
nohup bash experiments/4_analysis/run_analysis.sh r1 > r1.log 2>&1 &
nohup bash experiments/4_analysis/run_analysis.sh r2 > r2.log 2>&1 &
bash experiments/4_analysis/run_analysis.sh x1
bash experiments/4_analysis/run_analysis.sh figures
E2_CSV=$PROBE_ROOT/e2_metrics.csv DOSE_TABLE=$DATA_ROOT/controlled/obr_dose_data_table.csv \
  bash experiments/4_analysis/run_analysis.sh assets
```

The only dependency chain is `r0 -> pool -> r1 -> r2`. `SCALES=4B` restricts `pool`,
`r1` and `r2` to the 4B branch. `assets` can be run at any time; an analysis that has not
run is marked "not measured" rather than drawn as zero.

* `r0` pairs models prompt by prompt, runs the episode state machine and the two-stage
  competing-risk estimate for the nine contrasts, and standardizes exposure and persistence
  with 5,000 paired prompt resamples and a fixed common tail cutoff.
* `pool` cuts 256 anchors per scale from the seed-42 raw and cleaned responses (64 natural
  stop, 32 early, 32 before first reuse per source; at most one anchor per response) and
  splits them by prompt into development and test.
* `r1` scores every same-scale model on identical prefixes: first-token stop margin, close
  reachability after top-k/top-p, 16 sampled next events, EOS probability after a forced
  legal close.
* `r2` collects last-position residuals, fits the cleaned-minus-raw mean contrast on
  development anchors, selects the layer, calibrates the close-bias control on development
  anchors, runs the 13-condition single-pulse short readouts (3 strengths, 8 equal-norm
  random directions, close bias) and, at 4B, the 3,072 full continuations with repetition
  and retained-content scoring. At 8B it reads the contrast in SAE feature coordinates.
* `x1` builds 512 artificial complete-motif probes (128 prompts x m in {1,2,4,8}) and
  scores repeat and recovery candidates under four 4B models.

### Step 5 - input-support probe (GPU + CPU)

```bash
bash experiments/5_support_probe/run_support_probe.sh --build    # CPU, about 2 minutes
nohup bash experiments/5_support_probe/run_support_probe.sh > probe.log 2>&1 &
python experiments/5_support_probe/e2_diagnose.py --result_root $PROBE_ROOT --sections gates
```

384 anchors and 2,560 inputs per model, the 18 models of the appendix table by default,
first-token stop margin only (no sampled continuations), 2,000 anchor-bootstrap resamples.

## 5. Checking results without 8 GPUs

* `bash run_tests.sh` exercises the parser, the episode state machine, the competing-risk
  estimator and its standardization, the orbit detector, F1 scoring, the decoding-policy
  arithmetic (presence penalty, temperature, top-k/top-p), anchor construction, direction
  fitting and layer selection, pulse construction, motif probes, the support-probe builder
  and analysis, the bootstrap, and the completeness checks of the drivers.
* **All six figures of the paper regenerate from the shipped results in seconds:**
  `python experiments/4_analysis/make_figures.py --results reference_results --output out/figures --previews out/figures`
* Steps 1, `r0`, `pool`, `assets` and `figures` of step 4, and the anchor build of step 5 are CPU
  only. Given the step-3 outputs of two models, `r0` reproduces the corresponding
  contrast of Figure 2 on a laptop.
* A minimal GPU replication of the main claim needs two training runs and two evaluations
  (`qwen3-4b-cleanv2-s42`, `qwen3-4b-keep4-s42`), then `SCALES=4B` for `pool`, `r1`, `r2`.

**Expected variation.** Data construction, anchor selection, direction fitting, all
bootstrap intervals and the short/long sampling schedules are seeded and deterministic
given their inputs. Training and vLLM generation are seeded but not bit-reproducible across
hardware, library versions or batch composition, so retrained models will differ from ours
at the level of sampling noise; the two 4B training seeds in the paper indicate its size.

## 6. Release scope

Paths are configured through `env.sh` and default to directories within the
repository. Training, evaluation and support probes share `tcr/prompt_template.py`;
its rendered prompt matches the released SWIFT records. Comments do not form part
of that prompt. The run matrix evaluates final checkpoints only.

Reference results include aggregate measurements and the example used in Figure
1(b). That example is supplied as a fixed result file. Model checkpoints, raw
generation outputs, tokenizers, optional upstream data and the input-edit
construction pipeline are not included. No full GPU rerun is certified by the
CPU tests. GPU dependencies in `requirements-gpu.txt` are version constraints,
not a complete lock of a validated training environment.

## 7. License

MIT (see `LICENSE`). Data terms are documented in the
[full dataset card](https://huggingface.co/datasets/tcr-repro-2027/termination-control-data).
Third-party source-document excerpts in the reference results are not relicensed
by the code license; consult the dataset card for their applicable terms.
