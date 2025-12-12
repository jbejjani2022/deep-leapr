# RM Generalization Experiments

This folder holds scripts to compare a LeaPR text-regression model vs a baseline neural reward model (GPT-2 helpful RM) on multiple preference datasets.

## Datasets (pairwise)
- `hh_rlhf`: Anthropic/hh-rlhf, helpful-base (test split)
- `oasst1`: tasksource/oasst1_pairwise_rlhf_reward (train split)
- `pku_saferlhf`: PKU-Alignment/PKU-SafeRLHF (train split)

All are mapped to a common `{prompt, chosen, rejected}` schema via `datasets.yaml`. Default sample cap is 2000; override with `--limit`.

## Prereqs
```bash
pip install -r requirements.txt
pip install transformers datasets accelerate textstat spacy textblob nltk vaderSentiment
python -m spacy download en_core_web_sm
```
(Those extra NLP deps are needed because the LeaPR expert features may call them.)

## Run evaluation
```bash
python experiments/rm-generalization/eval_pairwise.py \
  --leapr-model results/models/combo__text_regression_rm_helpful__gpt-4o-mini.pkl \
  --leapr-api-level expert \
  --datasets hh_rlhf oasst1 pku_saferlhf \
  --limit 2000 \
  --output-root experiments/rm-generalization/runs
```

Outputs per dataset:
- CSV with per-sample scores and correctness for both models
- Summary JSON with accuracies and score stats
- A combined `summary.json` across datasets

## Notes
- The GPT-2 baseline is `Ray2333/gpt2-large-helpful-reward_model`.
- LeaPR scoring runs your saved RF model; if it wraps `RFValueFunction`, features are taken from the checkpoint, otherwise looked up by stem in `results/features/`.
- If you add a new dataset, edit `datasets.yaml` to supply `hf_id`, `split`, and field names for `chosen`/`rejected` (and optional `prompt`).
