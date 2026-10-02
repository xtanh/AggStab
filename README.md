# AggStab: Preference-aligned inverse folding for joint optimization of aggregation resistance and folding stability

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)
![ProteinMPNN](https://img.shields.io/badge/generator-ProteinMPNN-66558C)
![Status](https://img.shields.io/badge/status-pretrained%20inference-6E9B78)

AggStab is a fixed-backbone inverse-folding framework for jointly optimizing
resistance to stress-induced aggregation and folding stability. It combines an
aligned ProteinMPNN policy with two frozen, backbone-conditioned SaProt reward
models.

![AggStab overview](assets/aggstab_overview.png)

This repository is a clean, inference-focused release. It supports:

- aggregation-resistance prediction on a fixed backbone;
- folding-stability prediction on the learned
  $\Delta G_{\mathrm{unfolding}}$ scale;
- generation and joint reranking of aggregation-resistant, stable sequences.

Training pipelines, datasets, experiment records, manuscript sources, and
internal analysis scripts are intentionally not included.

## Installation

```bash
git clone https://github.com/xtanh/AggStab.git
cd AggStab
conda env create -f environment.yml
conda activate aggstab
```

AggStab uses the official
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN) implementation for
sequence generation and Foldseek for backbone 3Di tokens:

```bash
git clone https://github.com/dauparas/ProteinMPNN.git third_party/ProteinMPNN
export PROTEINMPNN_DIR="$PWD/third_party/ProteinMPNN"
export PROTEINMPNN_CKPT="$PROTEINMPNN_DIR/vanilla_model_weights/v_48_020.pt"
export FOLDSEEK_BIN=/absolute/path/to/foldseek
export SAPROT_MODEL_PATH=/absolute/path/to/SaProt_650M_PDB
```

Download `SaProt_650M_PDB` following the official
[SaProt repository](https://github.com/westlake-repl/Saprot). The current
interface expects a single-chain PDB backbone and sequences of the same length
as the selected chain.

## Pretrained models

Download the versioned AggStab model archive and place its files as follows:

```text
checkpoints/
  aggregation_reward.ckpt
  stability_reward.ckpt
  aggstab_policy.pt
```

**[Download AggStab v1.0.0 pretrained models from Google Drive](https://drive.google.com/drive/folders/1GWx6yQpw3r3cLA0EUXrK5cvZ5A1CoenB)**

The Google Drive folder contains the three files shown above. Alternatively,
download the complete folder with `gdown`:

```bash
python -m pip install gdown
gdown --folder \
  'https://drive.google.com/drive/folders/1GWx6yQpw3r3cLA0EUXrK5cvZ5A1CoenB' \
  -O checkpoints
```

The reward checkpoints are kept outside Git because each is approximately
2.65 GB. Exact file sizes, roles, and SHA256 checksums are listed in
[`models/model_manifest.yaml`](models/model_manifest.yaml).

## Predict aggregation resistance

Score the sequence encoded by a target PDB:

```bash
python scripts/predict_properties.py \
  --pdb /path/to/target.pdb \
  --chain A \
  --aggregation_ckpt checkpoints/aggregation_reward.ckpt \
  --output_csv outputs/target_aggregation.csv \
  --device cuda:0
```

Add `--sequence SEQUENCE` to score one design or `--fasta candidates.fasta`
to score multiple designs on the same backbone. Larger
`aggregation_resistance_score` values indicate stronger predicted resistance
to stress-induced aggregation. `aggregation_gain_vs_wt` is relative to the
PDB-encoded sequence.

## Predict folding stability

```bash
python scripts/predict_properties.py \
  --pdb /path/to/target.pdb \
  --chain A \
  --fasta candidates.fasta \
  --stability_ckpt checkpoints/stability_reward.ckpt \
  --output_csv outputs/target_stability.csv \
  --device cuda:0
```

Larger `predicted_deltaG` values indicate stronger predicted folding
stability. `stability_gain_vs_wt` is the design prediction minus the
prediction for the PDB-encoded sequence. These values are computational
predictions on the model's training-assay scale, not direct experimental
measurements.

Supply both checkpoints in one command to obtain both properties:

```bash
python scripts/predict_properties.py \
  --pdb /path/to/target.pdb \
  --fasta candidates.fasta \
  --aggregation_ckpt checkpoints/aggregation_reward.ckpt \
  --stability_ckpt checkpoints/stability_reward.ckpt \
  --output_csv outputs/target_properties.csv \
  --device cuda:0
```

## Design sequences with AggStab

Generate 48 sequences, score them with both rewards, apply the release's staged
quality filters, and retain the top three:

```bash
python scripts/design_with_aggstab.py \
  --pdb /path/to/target.pdb \
  --chain A \
  --policy_ckpt checkpoints/aggstab_policy.pt \
  --aggregation_ckpt checkpoints/aggregation_reward.ckpt \
  --stability_ckpt checkpoints/stability_reward.ckpt \
  --num_samples 48 \
  --temperature 0.5 \
  --top_k 3 \
  --output_dir outputs/target_design \
  --device cuda:0
```

Outputs:

```text
outputs/target_design/
  all_candidates.csv
  top3_candidates.csv
  top3_candidates.fasta
  run_config.json
```

Candidates are deduplicated before scoring. The first filtering stage that
contains at least `top_k` candidates supplies the complete retained set;
stages are not accumulated. Structural prediction and experimental validation
are recommended before synthesis.

For AlphaFold PDB files whose B-factor field stores pLDDT, add
`--mask_low_confidence` to mask low-confidence 3Di tokens. Do not use this
flag for experimental B-factors.

## Repository layout

```text
assets/                         Overview figure
configs/                        Released reward-model configurations
models/model_manifest.yaml      Checkpoint metadata and checksums
scripts/predict_properties.py   Property prediction CLI
scripts/design_with_aggstab.py  Sequence-design CLI
src/inference.py                Backbone extraction and reward inference
src/models/                     Released reward architecture
src/mpnn/                       ProteinMPNN inference wrapper
```

## Scope and limitations

- The release currently supports single-chain fixed-backbone inference.
- Reward scores are model predictions and do not replace structural or
  experimental validation.
- Large checkpoints, generated structures, trajectories, datasets, and paper
  analysis files are not distributed in this repository.

## Acknowledgements

AggStab builds on
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN),
[SaProt](https://github.com/westlake-repl/Saprot), and Foldseek. Please cite
the original methods and comply with their respective licenses.

## License

No license has yet been assigned to the original AggStab code. Until a license
is added, contact the authors for permission to reuse or redistribute it.

## Citation

Citation information will be added when the manuscript becomes publicly
available.
