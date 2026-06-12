# DyME: Empowering Small-scale VLMs with Reliable Thinking Capabilities

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](#)
[![arXiv](https://img.shields.io/badge/arXiv-2506.23061-b31b1b.svg)](https://arxiv.org/abs/2506.23061)

This repository provides the official implementation of **DyME** (**Dy**namically selecting between **M**emorization and **E**xploration), accepted at **ICLR 2026**.

## Overview

Small-scale Vision-Language Models (SVLMs) are highly suited for proprietary tasks, but equipping them with reasoning and thinking capabilities remains challenging. Traditional Supervised Fine-Tuning (SFT) can force memorization of pseudo thinking traces, while Reinforcement Learning with Verifiable Reward (RLVR) often leads to unstable exploration (advantage collapse) due to limited model capacity.

**DyME** is a novel training paradigm that dynamically synergizes SFT and RLVR. At each optimization step, DyME dynamically selects between Memorization (via SFT) and Exploration (via RLVR), ensuring every update contributes to an optimal trade-off. To further enhance this, we introduce a **Visual Supervision mechanism** (a visual checker and refiner) to inject dynamically enhanced, image-grounded guidance during training. 

Extensive experiments show that DyME delivers substantial performance improvements, establishing it as a robust strategy for stabilizing SVLM learning.


## Repository Structure

```text
DyME/
├── client_utils/         # Client tools for online Visual Supervision (LLM API)
├── data/                 # Preprocessed textual datasets
├── data_utils/           # Data processing and formatting scripts
│   ├── aokvqa/
│   ├── chart/
│   └── commom_util.py
├── eval/                 # Evaluation scripts for different benchmarks
├── reward_utils/         # Reward function implementations for RLVR
├── config/               # Modular configuration files for experiments
├── opsd_utils/           # Privileged-context OPSD / TriMode extensions for DyMETrainer
├── default_config.yaml   # Default DDP (MULTI_GPU, no DeepSpeed required)
├── default_config_deepspeed.yaml  # Optional ZeRO-2 offload when deepspeed is installed
├── main.py               # Entry point for DyME training
├── main_*.py             # Additional experimental variants (e.g., 7B, LLM-only)
├── requirements.txt      # Python dependencies
└── ...
```


## Configuration

Before launching training, please prepare the relevant configuration files. The main settings are managed through configuration files such as `config/config.py` and `default_config.yaml`.

### `CLIENT_CONFIG`

This configuration is required when **Visual Supervision** is enabled. It specifies the online large-model API used by the visual checker and visual refiner.

### `TRAINING_CONFIG`

This section contains standard training hyperparameters for both the memorization phase and the exploration phase, including optimizer settings, batch size, learning rate, and related options.

### `RL_CONFIG`

This section defines critical variables for reward computation and response parsing during RLVR training. In particular, the following delimiters must be properly specified:

* `answer_flag`: used to explicitly separate the final answer from auxiliary generated content such as intermediate reasoning traces.
* `end_flag`: used to mark the end of generation.

These delimiters are essential for stable parsing, reward assignment, and evaluation consistency.

### `DYME_OPSD_CONFIG` (OPSD / TriMode)

`config/config.py` defines `DYME_OPSD_CONFIG`, merged into `CONFIG["opsd"]`. When `enabled=False` (default), training follows the original DyME behavior. Set `enabled=True` or pass CLI flags to activate privileged-context **Self-OPSD** inside `DyMETrainer`.

| Field | Description |
| --- | --- |
| `enabled` | Master switch. `False` → original DyME only. |
| `mode` | Routing mode (see table below). |
| `privileged_profile` | Teacher preset: `text` \| `visual` \| `hybrid` (default **`hybrid`** in `config_trimode.py`). |
| `privileged_providers` | Override provider list; default derived from profile. |
| `privileged_image` | Teacher image layout: `mode` `single` (ChartQA default) or `dual` (full + crop); plus `crop_strategy`, `bbox_coord`, `margin_ratio`. |
| `privileged_debug` | Periodic artifact logging: `save_images`, `image_subdir` (`logs/images`), `max_samples_per_detail`. |
| `gate.correct_threshold` | Reward threshold to count a rollout as correct. |
| `gate.teacher_recoverable` | Recoverability gate: `privileged_available` (default) or `logprob_gain`. |
| `loss.beta` | JSD temperature for OPSD distillation. |
| `loss.opsd_weight` / `grpo_weight` / `sft_weight` | Per-mode loss weights. |

**Routing modes (`mode`):**

| Mode | Behavior |
| --- | --- |
| `dyme` | Original DyME: any correct rollout → GRPO; all wrong → SFT. |
| `trimode` | Any correct → OPSD (replaces GRPO); all wrong → SFT (DyME cold-start via `sft_check`, ignores recoverable). |
| `opsd_only` | All prompts use OPSD. |
| `replace_sft` | Any correct → GRPO; all wrong → OPSD (no SFT). |
| `opsd_on_wrong` | Any correct → GRPO; all wrong + recoverable → OPSD; all wrong + not recoverable → SFT (legacy three-way routing). |
| `grpo_opsd_joint` | Any correct → GRPO (+ optional joint OPSD loss); all wrong + recoverable → OPSD; else SFT. |

Under `trimode`, the SFT share is determined by accuracy (how often prompts are all-wrong) and DyME's per-group `sft_check` (teacher injection on the first generation only)—no extra `sft_ratio` hyperparameter.

**Privileged profiles** (`privileged_profile`):

| Profile | Teacher images | Teacher text suffix |
| --- | --- | --- |
| `text` | Single full image (same as student) | hint + answer |
| `visual` | **Dual**: full + evidence crop | Visual Facts only (no answer leak) |
| `hybrid` | Single full image by default (`privileged_image.mode=single`); dual with `mode=dual` | Visual Facts + hint + answer |

Student `collate_fn` never reads privileged fields. With `privileged_image.mode=dual`, teacher forward uses `[full, crop]`; crop comes from normalized `evidence_bbox` (C2), A-OKVQA `visual_fact` heuristic (D2), or center fallback (D1). ChartQA defaults to `single` (no crop zoom).

**Privileged providers** (under `opsd_utils/privileged/`):

* `text` — uses the `hint` / `answer` fields in training samples.
* `visual_facts` — uses `visual_fact` JSON (B1 raw string), plus ChartQA `visual_fact_hint` (F1) and `visual_fact_deplot` (F2).
* `crop` — evidence region as second teacher image (via `image_utils`, not a text suffix).
* `hybrid` — combines text + visual_facts providers per profile.

**Debug / artifact logging**

* Verbose OPSD logs: `--opsd_debug` or `DYME_OPSD_DEBUG=1`.
* Full diagnostic bundle every N steps: `--opsd_detail_every N` or `DYME_OPSD_DETAIL_EVERY`.
* On detail steps, teacher privileged images are saved under `{output_dir}/logs/images/` as `step_XXXXXX_idx_Y_full.png`, `_crop.png`, and `_meta.json` (controlled by `privileged_debug.max_samples_per_detail`).

**ChartQA visual-facts preprocessing (run on server before TriMode training)**

TriMode with `privileged_providers=text,visual_facts` requires `visual_fact_hint` / `visual_fact_deplot` (and optionally `visual_fact`) on each sample. Raw `train_medium.json` only has `hint` — without this step, logs show `visual_fact_len=0` and the VisualFacts teacher channel is empty.

From the repo root on your GPU server:

```bash
cd /path/to/agentic-rl-main   # project root (parent of scripts/, config/, data/)

# F1: copy hint → visual_fact_hint (+ visual_fact for backward compat)
python scripts/build_visual_facts_chartqa.py \
  --input data/chartqa/train_medium.json \
  --output data/chartqa/train_medium_vf_hint.json \
  --also-set-visual-fact

# F2: DePlot offline table extraction (google/deplot, batched GPU inference; default ON)
python scripts/build_visual_facts_chartqa_deplot.py \
  --input data/chartqa/train_medium_vf_hint.json \
  --output data/chartqa/train_medium_vf_full.json \
  --batch-size 8 \
  --cache data/chartqa/deplot_cache.json

# Fast placeholder-only mode (no GPU / CI): add --no-enabled
# DYME_DEPLOT_ENABLED=0 bash scripts/train_local_gpus.sh

# quick sanity check (expect non-zero lengths)
python -c "
import json, random
d = json.load(open('data/chartqa/train_medium_vf_full.json', encoding='utf-8'))
s = random.choice(d)
assert s.get('visual_fact_hint'), 'missing visual_fact_hint'
assert s.get('visual_fact_deplot'), 'missing visual_fact_deplot'
print('ok', len(d), 'records; sample visual_fact_hint len', len(s['visual_fact_hint']))
"
```

`config/config.py` points `train_dataset` at `data/chartqa/train_medium_vf_full.json`. Generated `*_vf_*.json` files are gitignored — **generate them on each server** (or copy from shared storage); do not rely on cloning them from GitHub.

`scripts/train_local_gpus.sh` will auto-run the two Python steps above if `train_medium_vf_full.json` is missing.

**Training examples (TriMode + hybrid default)**

```bash
# Text-only OPSD ablation
python main.py --config trimode --opsd_privilege_profile text

# Vision-OPD style (no answer text to teacher)
python main.py --config trimode --opsd_privilege_profile visual

# Full hybrid (default in config_trimode)
python main.py --config trimode --opsd_privilege_profile hybrid --opsd_detail_every 10
```

**Privileged sample schema**

| Field | Used by | Notes |
| --- | --- | --- |
| `prompt`, `image` | Student + teacher | Student always single full image |
| `hint`, `answer` | Teacher (`text` / `hybrid`) | Never in student collate |
| `visual_fact` | Teacher | Raw JSON string (A-OKVQA) |
| `visual_fact_hint` | Teacher (ChartQA F1) | Hint placeholder pipeline |
| `visual_fact_deplot` | Teacher (ChartQA F2) | DePlot `parsed_table` text (`google/deplot`; placeholder skipped) |
| `evidence_bbox` | Teacher crop | Normalized `[x0,y0,x1,y1]` in `[0,1]` |

Adapter helpers for future datasets: `data_utils/privileged_schema.py` (`normalize_evidence_bbox`, `parse_visual_fact`, `resolve_crop_bbox`).

For legacy ChartQA single-field preprocessing, see `scripts/build_visual_facts_chartqa.py`.



## Data Preparation

We provide example preprocessing scripts in the `data_utils/` directory. After preprocessing, the training data should be organized as a list of dictionaries (e.g., `metadata_list`) following the format below:

```python
metadata_list.append({
    "question": question,               # Full prompt used for training
    "question_wo_prompt": question,     # Raw question without prompt template
    "answer": answer,                   # SFT target; should follow the answer_flag format
    "image": image_save_path,           # Local path to the corresponding image
})
```

### Field Description

* `question`: the complete model input used during training.
* `question_wo_prompt`: the raw question content without any additional prompt wrapper.
* `answer`: the training target for SFT; this field should be formatted consistently with the delimiter specification in `RL_CONFIG`.
* `image`: the local file path of the associated image, if applicable.



## Environment Setup

Please first install the required dependencies and configure the distributed training environment:

```bash
pip install -r requirements.txt
accelerate config
```

The `accelerate config` step is required to initialize the distributed environment for both training and evaluation.



## Dataset Setup

### Text Data

Preprocessed text splits are provided under the `data/` directory.

### Image Data

Due to storage constraints, image datasets are not included in this repository. Download scripts write images under `data/images/` by default:

```text
data/images/
├── chartqa/
│   ├── images/     # train_000000.png, val_000000.png, test_000000.png, ...
│   └── json/       # train.json, val.json, test.json (from download.py)
└── aokvqa/
    ├── images/     # train_0000000.png, ...
    └── json/       # train.json, validation.json, test.json (from download.py)
```

**ChartQA** (images only, no API required):

```bash
python data_utils/chart/download.py
```

**A-OKVQA** (images only by default; set `FETCH_VISUAL_FACTS=1` only if local VLM APIs are running on ports 23333–23340):

```bash
python data_utils/aokvqa/download.py
```

If you already downloaded ChartQA to `chartqa_output/` at the project root, move it into the canonical layout:

```bash
mkdir -p data/images
mv chartqa_output data/images/chartqa
```

Preprocessed text annotations with hints live separately under `data/chartqa/` and `data/aokvqa/`. Image paths inside those JSON files are resolved automatically at load time (legacy prefixes like `/chartqa_output/` map to `data/images/chartqa/`).

### Demo Samples

A small subset of demo images for verifying the data loading pipeline may be provided in a future update.



## Dataset Examples

### ChartQA

**ChartQA** is a visual question answering benchmark grounded in chart images. To illustrate different supervision granularities, we provide representative examples with three levels of reasoning-trace quality: **High**, **Medium**, and **Low**.

<div align="center">

| Example                                                         |
| --------------------------------------------------------------- |
| <img src="figs/chartqa.png" alt="ChartQA Example" width="220"/> |

</div>

#### High-quality Example

<details>
<summary><code>High-quality ChartQA Example</code></summary>

```json
{
  "question": "When does the unfavorable view reach the peak?",
  "answer": "2017",
  "hint": "<SUMMARY> To solve the problem, I will examine the image to identify trends in unfavorable views of Pakistan in India over time. I'll closely inspect the data points within the graph to determine the year where the \"very unfavorable view\" reaches its peak. This involves identifying the maximum value on the vertical axis and noting the corresponding year on the horizontal axis. </SUMMARY> \n\n<CAPTION> The image is a line graph titled \"Very unfavorable views of Pakistan increasing in India,\" with the subtitle \"Very unfavorable view of Pakistan.\" The y-axis represents the percentage of unfavorable views, ranging from 0% to 100%. The x-axis displays years from 2013 to 2017. The data points show the percentages of very unfavorable views over these years, with specific values marked: 54% in 2013, 49% in 2014, 51% in 2015, 55% in 2016, and 64% in 2017. The graph shows a general upward trend in unfavorable views, peaking in 2017. </CAPTION> \n\n<REASONING> To determine when the unfavorable view reaches its peak, one should observe the graph for the data point with the highest percentage on the y-axis. The graph shows percentages for each year from 2013 to 2017: starting at 54% in 2013, decreasing to 49% in 2014, and then gradually increasing to 51% in 2015 and 55% in 2016. The graph culminates with the highest percentage of 64% in 2017. Thus, the peak of unfavorable views is associated with the year 2017. </REASONING> \n\n<CONCLUSION> 2017 </CONCLUSION>"
}
```

</details>

#### Medium-quality Example

<details>
<summary><code>Medium-quality ChartQA Example</code></summary>

```json
{
  "question": "When does the unfavorable view reach the peak?",
  "answer": "2017",
  "hint": "Goal: Find the year when the unfavorable view reaches its peak.\nObservation: The data shows the values for each year are: 2013: 0, 2014: 0, 2015: 0, 2016: 55, and 2017: 64.\nReasoning: By comparing the values in each year, the highest value is 64, which occurs in 2017.\nConclusion: The unfavorable view reaches its peak in 2017."
}
```

</details>

#### Low-quality Example

<details>
<summary><code>Low-quality ChartQA Example</code></summary>

```json
{
  "question": "When does the unfavorable view reach the peak?",
  "answer": "2017",
  "hint": "I'm trying to figure out the year when the unfavorable view reaches its highest point. Looking at the data, I see that the values for each year are pretty low until 2016, where it jumps to 55. But the peak doesn't happen until 2017, when the value spikes to 64. So, it seems like the unfavorable view really hits its maximum in 2017."
}
```

</details>


### A-OKVQA

**A-OKVQA** is an open-ended visual question answering benchmark that requires world knowledge, commonsense reasoning, and visual understanding. Below we provide a representative example together with its corresponding annotation.

<div align="center">

| Example                                                        |
| -------------------------------------------------------------- |
| <img src="figs/aokvqa.png" alt="A-OKVQA Example" width="220"/> |

</div>

<details>
<summary><code>View A-OKVQA JSON Example</code></summary>

```json
{
  "question": "What is the man by the bags awaiting?",
  "answer": "cab",
  "visual_fact": "{\n  \"description\": \"The image shows a man standing in the middle of a street, facing away from the camera. He is holding a red bag in one hand and appears to be pulling a black suitcase with wheels. Another black suitcase is lying on the ground near him. The setting is an urban area with houses, parked cars, and trees in the background. The man seems to be waiting or preparing to cross the street.\",\n  \"objects\": [\n    {\n      \"name\": \"man\",\n      \"attributes\": [\"wearing a light blue and white shirt\", \"blue jeans\", \"carrying a red bag\", \"pulling a black suitcase\"],\n      \"position\": \"center\"\n    },\n    {\n      \"name\": \"red bag\",\n      \"attributes\": [\"held by the man\"],\n      \"position\": \"left side of the man\"\n    },\n    {\n      \"name\": \"black suitcase\",\n      \"attributes\": [\"with wheels\", \"being pulled by the man\"],\n      \"position\": \"near the man's feet\"\n    },\n    {\n      \"name\": \"black suitcase\",\n      \"attributes\": [\"on the ground\"],\n      \"position\": \"on the ground near the man\"\n    },\n    {\n      \"name\": \"street\",\n      \"attributes\": [\"asphalt\", \"urban setting\"],\n      \"position\": \"foreground\"\n    },\n    {\n      \"name\": \"houses\",\n      \"attributes\": [\"visible in the background\"],\n      \"position\": \"left side\"\n    },\n    {\n      \"name\": \"parked cars\",\n      \"attributes\": [\"red SUV\", \"other vehicles\"],\n      \"position\": \"left and center background\"\n    },\n    {\n      \"name\": \"trees\",\n      \"attributes\": [\"green foliage\"],\n      \"position\": \"right side\"\n    }\n  ]\n}",
  "hint": "A train would not be on the street, he would not have luggage waiting for a delivery, and the skateboarder is there and not paying attention to him, so a cab is the only plausible answer."
}
```

</details>



### GSM8K

**GSM8K** is a mathematical word problem benchmark. Since it is text-only, we provide a representative question-answer example together with its reasoning trace.

<details>
<summary><code>View GSM8K JSON Example</code></summary>

```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "72",
  "hint": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

</details>



## Training

All training scripts are launched using `accelerate`. Pass `--config` as a **Python config file path** (recommended) or a shorthand alias (`norm`, `trimode`, `llavacot`, `low`, `aok`).

**Important:** `num_processes` must match the number of visible GPUs on your node. Helper scripts auto-detect GPU count and pick a launch config:

- **Default:** `default_config.yaml` → native **DDP** (`MULTI_GPU`), **no DeepSpeed install required** (sufficient for 0.5B on multi-GPU).
- **Optional:** if `deepspeed` is installed, scripts auto-select `default_config_deepspeed.yaml` (ZeRO-2 + CPU offload). Force DDP with `ACCELERATE_CONFIG=default_config.yaml`.

```bash
# 2-GPU node example (DDP)
accelerate launch --config_file default_config.yaml --num_processes 2 main.py --config config/config.py --mode rl

# 8-GPU node example
accelerate launch --config_file default_config_8gpu.yaml --num_processes 8 main.py --config config/config.py --mode rl
```

Or override explicitly: `NUM_GPUS=4 bash scripts/train_trimode.sh`

For TriMode on **all visible local GPUs** (auto-detect via `CUDA_VISIBLE_DEVICES` / `torch.cuda.device_count()`):

```bash
# 1) One-time (or when raw data changes): enrich ChartQA JSON on the server — see
#    "ChartQA visual-facts preprocessing" above. train_local_gpus.sh also auto-runs
#    this if train_medium_vf_full.json is absent.

# 2) Start training (default: OPSD verbose off, detail every 50 steps, probe on)
bash scripts/train_local_gpus.sh

# Optional: full verbose debug (large logs)
# DYME_OPSD_DEBUG=1 DYME_OPSD_DETAIL_EVERY=10 bash scripts/train_local_gpus.sh

# Roll back to original trimode config (pre-antidegen hyperparameters)
# DYME_CONFIG=config/config_trimode.py bash scripts/train_local_gpus.sh
```

### Anti-degeneration config (`config_trimode_antidegen`)

`scripts/train_local_gpus.sh` defaults to **`config/config_trimode_antidegen.py`** (alias `trimode_antidegen`). Overrides are based on offline analysis of `train_trimode_4gpu_20260610_173637.log` (1225 steps):

| Issue | Baseline log evidence | Antidegen change |
|-------|----------------------|------------------|
| Logit collapse | `LOGIT_MODE_COLLAPSE` 212×; step 1 clip 0→1.0; step 1175 clip≈0.92 | `max_completion_length=150`, `temperature=0.7`, `repetition_penalty=1.25` |
| Step-1 gradient shock | `GEN_CLIP_COLLAPSE` from step 1; `OPT_GRAD_SPIKE` 44× | `learning_rate=5e-5`, `warmup_steps=50` |
| OPSD coverage low | `opsd_mask` mean 5.6%; 492/1226 zero-mask steps | `require_format_for_opsd=False` (env default `DYME_OPSD_REQUIRE_FORMAT=0`) |
| RL signal sparse | `RL_ZERO_SIGNAL` expected in trimode | `reward_weights=[0.5, 1.5, 1.0]` (format, context F1, acc) |
| visual_fact empty | `visual_fact_empty_rate=0` throughout | no data change |

Environment overrides:

```bash
export DYME_CONFIG=config/config_trimode_antidegen.py   # default in train_local_gpus.sh
export DYME_OPSD_REQUIRE_FORMAT=0                       # antidegen default; set 1 to restore strict gate
export DYME_REWARD_WEIGHTS=0.5,1.5,1.0                  # format, context, accuracy
```

After a new run (~200+ steps), compare against the baseline log:

```bash
python scripts/parse_trimode_log.py outputs/logs/train_trimode_*_new.log
python scripts/degeneration_report.py outputs/logs/train_trimode_*_new.log
python scripts/compare_trimode_logs.py train_trimode_4gpu_20260610_173637.log outputs/logs/train_trimode_*_new.log
```

Success criteria (candidate vs baseline): step 1 `clip` &lt; 1.0; `LOGIT_MODE_COLLAPSE` count down &gt;30%; `opsd_mask` mean &gt; 8%; step 200+ `mean_length` median &lt; 130. `RL_ZERO_SIGNAL` may remain high (trimode design).


### 1. Training DyME (original)

Default config keeps OPSD disabled (`DYME_OPSD_CONFIG.enabled=False`):

```bash
accelerate launch main.py --config config/config.py --mode rl
```

### OPSD debug logging + tee

When debugging OPSD / TriMode (e.g. NCCL timeout), enable verbose logs and save stdout/stderr:

```bash
export DYME_OPSD_DEBUG=1
mkdir -p ./outputs/logs
LOG_FILE=./outputs/logs/train_$(date +%Y%m%d_%H%M%S).log

accelerate launch --config_file default_config.yaml --num_processes "$(nvidia-smi -L | wc -l)" main.py \
  --config config/config_trimode.py \
  --mode rl \
  --opsd_enabled \
  --opsd_debug \
  --opsd_mode trimode \
  --opsd_providers text,visual_facts \
  2>&1 | tee "${LOG_FILE}"
```

Logs are prefixed with `[OPSD-DEBUG]` and include rank, step, `[SYNC_POINT]` markers before every distributed collective in the OPSD chain (reward gather, teacher prompt build, metrics gather, OPSD loss). Search the log for the last `[SYNC_POINT]` on each rank to locate where a hang occurred.

You can also use the helper script (debug + tee enabled by default):

```bash
bash scripts/train_trimode.sh
```

Disable debug when not needed: `DYME_OPSD_DEBUG=0 bash scripts/train_trimode.sh`

### Periodic weak-signal diagnostics (`[OPSD-DETAIL]`)

Separate from per-step `[OPSD-DEBUG]` spam: every **N global steps** (default **10**, rank 0 only) a full diagnostic bundle is printed to investigate **reward ≈ 0** and **gradient ≈ 0** while the OPSD chain still runs:

- Generation: EOS rate, clipped ratio, effective completion tokens, decoded samples
- Reward: format / acc / context breakdown, advantage stats, per-sample table
- Routing: OPSD mask ratio, TriMode counts, advantage token distribution
- Loss: GRPO per-token logps, coef\_1, clip counts, weak-signal hints
- OPSD JSD: per-token JSD, student/teacher top-1 agreement, max-JSD token

Configure via config, CLI, or env:

```bash
# default: every 10 steps (config_trimode.py)
export DYME_OPSD_DETAIL_EVERY=10

python main.py --config config/config_trimode.py --mode rl \
  --opsd_enabled --opsd_detail_every 10

# disable periodic detail
export DYME_OPSD_DETAIL_EVERY=0
```

Search logs for `[OPSD-DETAIL]` (not `[OPSD-DEBUG]`).

**Per-generate probe (`[OPSD-PROBE]`)** — enabled by default in `config_trimode.py`; fires on every `(re)generate` on rank 0 (no need to wait for step 10). Logs raw `completion_ids`, decode with/without special tokens, `eos_idx`, flags `ONE_TOKEN` / `EMPTY_DECODE` / `FIRST_IS_EOS`, and patterns `PAREN_THEN_EOS` / `REPEAT_LOOP`. Disable with `DYME_OPSD_PROBE_ON_GENERATE=0` or `--no_opsd_probe_on_generate`.

**Deep generate debug (`[OPSD-GENDBG]`)** — runs alongside `[OPSD-PROBE]` when probe is enabled. Before each `model.generate`, logs model training context, prompt tail tokens/decode, and first-token logits (`p_eos`, `p_token_340`, `entropy`, `top5`) via **per-sample** forward (up to `probe_sample_count`, default 4) to avoid OOM on large VLM batches. After generate, logs greedy-vs-actual first token, delta vs previous regenerate, and cross-rank summary.

```bash
export DYME_OPSD_PROBE_ON_GENERATE=1   # default in config_trimode
grep -E '\[OPSD-(PROBE|GENDBG)\]' train.log
```

| Observation in logs | Likely root cause |
|---------------------|-------------------|
| `p_eos` very high + `greedy_token_id==eos` | Weight collapse / train-mode distribution |
| `prompt_tail_decode` ends with unclosed template + high `p_token_340` | Prompt / chat template issue (legacy `"Answer: .."` quoted placeholder biased token 340 `)`; fixed in `data_utils/rl_prompt.py`) |
| `greedy_matches_actual=False` with low `p_eos` | Sampling noise (temperature / top_p) |
| Large `one_token_count` gap across ranks in `cross_rank` | Data sharding / batch composition |
| `delta_one_token_count` spikes at `generate_call_index>=2` | Weight drift after optimizer step |

Optional env overrides:

```bash
export DYME_OPSD_PROBE_FIRST_TOKEN_LOGITS=0   # skip extra forward before generate
export DYME_OPSD_PROBE_PROMPT_TAIL_TOKENS=24
export DYME_OPSD_PROBE_LOG_MODEL_CONTEXT=0
```

### 2. Training TriMode (DyME + OPSD)

Use `config/config_trimode.py` (OPSD pre-enabled) or override on the base config via CLI:

```bash
accelerate launch main.py \
  --config config/config_trimode.py \
  --mode rl \
  --opsd_enabled \
  --opsd_mode trimode \
  --opsd_providers text,visual_facts
```

Equivalent one-liner with base config + CLI only:

```bash
accelerate launch main.py --config config/config.py --mode rl \
  --opsd_enabled --opsd_mode trimode --opsd_providers text,visual_facts
```

**CLI OPSD flags** (override `CONFIG["opsd"]`):

| Flag | Description |
| --- | --- |
| `--opsd_enabled` | Enable OPSD / TriMode extensions. |
| `--opsd_debug` | Verbose OPSD chain logs (`[OPSD-DEBUG]`, or env `DYME_OPSD_DEBUG=1`). |
| `--opsd_detail_every N` | Full weak-signal bundle every N steps (`[OPSD-DETAIL]`, default 10; `0` = off). |
| `--opsd_probe_on_generate` / `--no_opsd_probe_on_generate` | Per-generate `[OPSD-PROBE]` on rank 0 (trimode default on). |
| `--opsd_mode MODE` | Routing mode: `trimode` (legacy), `rlsd` (anti-leakage), `copsd_opd`, `dyme`, `opsd_only`, `replace_sft`, … |
| `--opsd_providers LIST` | Comma-separated providers, e.g. `text`, `format_only`, `visual_facts`. Empty = same-prompt OPD only. |

### 2b. RLSD / anti-leakage OPSD (recommended for ChartQA)

`trimode` routes OPSD on **correct** completions and injects gold answer into the teacher prompt (information leakage). Use **`rlsd`** instead:

- **Correct** → GRPO (on-policy self-learning, no privileged suffix)
- **Wrong** → same-prompt OPSD / OPD (no `[Reference Answer]` in teacher)
- **All-wrong group** → online SFT replace (DyME cold-start; no separate offline SFT phase)

```bash
bash scripts/train_rlsd_chartqa.sh
# or: --config config/config_rlsd_chartqa.py --opsd_mode rlsd --opsd_providers format_only
```

**Cross-model OPD (7B frozen teacher + 0.5B student):**

```bash
export DYME_TEACHER_DEVICE_MAP=cuda:0
bash scripts/train_opd_7b_chartqa.sh
# eval: CHECKPOINT_DIR=./outputs/opd-7b-chartqa/final_checkpoint bash scripts/run_eval_ablation.sh
```

Note: `main.py --mode rl --config config/config.py` uses **`dyme_args`** (not the unused `grpo_args` block in the same file). Pure GRPO baselines use `main_rebuttal.py`.

**Helper scripts** (under `scripts/`):

```bash
# TriMode on ChartQA (legacy; leakage risk on ChartQA)
bash scripts/train_trimode.sh

# Anti-leakage RLSD (recommended)
bash scripts/train_rlsd_chartqa.sh

# Ablation matrix: MODE=dyme|trimode|replace_sft|opsd_only|...
MODE=trimode DYME_OPSD_PROVIDERS=text,visual_facts bash scripts/train_baselines.sh

# Post-training eval (set CHECKPOINT_DIR)
CHECKPOINT_DIR=./outputs/trimode-chartqa/final_checkpoint bash scripts/run_eval_ablation.sh
```

### 3. Reproducing Baselines

To reproduce baseline settings such as standard SFT or RL training, use `main_rebuttal.py` and specify the desired mode through `--mode`.

#### Supervised Fine-Tuning (SFT)

```bash
accelerate launch main_rebuttal.py --config config/config.py --mode sft
```

#### Reinforcement Learning (GRPO / RL)

```bash
accelerate launch main_rebuttal.py --config config/config.py --mode grpo
```

### 4. Additional Experimental Variants

For specific experimental settings such as different model scales or architecture-specific ablations, please use the corresponding scripts:

* `main_7B.py`: experiments at the 7B scale
* `main_llm.py`: LLM-specific variants
* `main_change.py`: additional ablation settings


## Evaluation

We support multi-process evaluation through `accelerate`. Evaluation scripts are located in the `eval/` directory and can be launched as Python modules.

### General Usage

```bash
accelerate launch -m eval.<eval_script_name>
```

### Example: ChartQA Evaluation

```bash
accelerate launch -m eval.eval_chartqa
```

### Evaluation Setup

Before running evaluation, please open the corresponding evaluation script (for example, `eval_chartqa.py`) and modify the following fields as needed:

* `model_id`: the path or identifier of the checkpoint to be evaluated
* prompt templates: these should match the formatting used during training

Ensuring consistency between training and evaluation prompts is important for obtaining reliable results.

## Citation

If you find this repository useful in your research, please consider citing our paper:

```bibtex
@inproceedings{dyme2026,
  title={Empowering Small VLMs to Think with Dynamic Memorization and Exploration},
  author={Jiazhen Liu, Yuchuan Deng, Long Chen},
  booktitle={ICLR},
  year={2026},
}
```
