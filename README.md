# Legal Judgment Prediction — Shortcut Learning Ablation

Capstone project studying whether a legal judgment prediction (LJP) model trained on Indian court
cases (PREDEX dataset) actually reasons from case **facts**, or whether it is shortcutting on
spurious signals — statute citations, precedent references, ruling language — that leak the
outcome directly into the input text.

## Model

`HierarchicalLegalBERT` ([opennyai_ablation.py](opennyai_ablation.py)):

- **Encoder**: [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT), frozen except the last 3
  transformer layers + pooler.
- **Chunking**: documents are split into 512-token chunks (256 stride, max 16 chunks) since case
  text exceeds BERT's context window.
- **Aggregation**: a 2-layer bidirectional GRU over per-chunk `[CLS]` embeddings, followed by a
  linear classifier for the binary outcome label.

## Method

Each case is segmented into rhetorical roles (Facts, Arguments, Ratio, Statute Analysis,
Precedent, Ruling clauses, ...) using [OpenNyAI](https://github.com/OpenNyAI/Opennyai)'s
rhetorical-role classifier. From these roles, 11 input variants are built per case and run through
the frozen trained model:

1. **Layer 1 — Graduated ablation** (`v0_facts_only` → `v25` → `v50` → `v75` → `v100_full_spu`):
   starting from facts-only text, spurious content (statute/precedent/ruling sentences pooled
   together) is injected in increasing amounts. Measures the dose-response relationship between
   spurious content and prediction shift.
2. **Layer 2 — Role-specific ablation** (`v_sta_only`, `v_pre_only`, `v_ruling_only`, `v_arg_rat`):
   adds one role type at a time on top of facts to isolate which specific role drives the shift.
   `v_arg_rat` (arguments + ratio) serves as a lower-risk control.
3. **Layer 3 — Counterfactual** (`v_counterfactual`): swaps every section number in the facts to a
   neutral value ("Section 144"). If the prediction flips, the model is keying on the specific
   number rather than case substance.

"Shift rate" = the fraction of cases where a variant's prediction differs from the `v0_facts_only`
baseline for the same case.

## Results

Run on 1,346 held-out PREDEX test cases (`predex_ablation_results.json`, plotted in
[combined_ablation.png](combined_ablation.png)):

| Variant | Accuracy | Shift vs. facts-only | Avg. confidence |
|---|---|---|---|
| `v0_facts_only` | 50.1% | — | 0.681 |
| `v25` | 61.0% | 28.5% | 0.725 |
| `v50` | 66.4% | 35.0% | 0.746 |
| `v75` | 68.5% | 37.6% | 0.760 |
| `v100_full_spu` | 68.5% | 36.0% | 0.752 |
| `v_sta_only` | 70.4% | 37.8% | 0.782 |
| `v_pre_only` | 53.0% | 10.3% | 0.686 |
| `v_ruling_only` | 51.2% | 14.5% | 0.681 |
| `v_arg_rat` (control) | 54.5% | 19.9% | 0.725 |
| `v_counterfactual` | 50.4% | 0.7% | 0.682 |
| `v_full` | 67.9% | 34.0% | 0.759 |

**Takeaways:**

- On facts alone, the model performs at chance (50.1% on a binary task) — it is not making
  reliable predictions from case substance by itself.
- Injecting spurious content raises accuracy by ~18 points and flips ~36% of predictions, showing
  strong reliance on non-factual content.
- **Statute-analysis sentences are the dominant shortcut**: `v_sta_only` alone accounts for nearly
  all of the full-spurious effect (70.4% accuracy, 37.8% shift), far more than precedent or ruling
  sentences.
- The counterfactual test shows only a 0.7% shift rate — well under the 10% threshold used to
  call it a shortcut — so the model is *not* keying on the specific section number, only on the
  presence/style of statute-analysis content in general.
- `v_full` (all roles concatenated) tracks closely with `v100_full_spu`, confirming the
  constructed variants are a faithful approximation of real full-text behavior.

## Setup

See [setup_workstation.sh](setup_workstation.sh) for the full remote workstation bootstrap
(Python 3.8 venv, pinned `spacy`/`opennyai`/`pydantic` install order, HuggingFace model/data pull).
Locally:

```bash
pip install -r requirements.txt
python opennyai_ablation.py
```

Requires `predex_test_deduped.csv`, `best_hierarchical_model.pt`, and a local `InLegalBERT/`
checkpoint directory (pulled from the project's HuggingFace repos — not checked into git).

## Files

| File | Purpose |
|---|---|
| `opennyai_ablation.py` | End-to-end pipeline: segmentation, variant construction, inference, analysis, plotting |
| `predex_ablation_results.json` | Per-case predictions/confidence for all 11 variants (1,346 cases) |
| `combined_ablation.png` | Three-panel figure: dose-response curve, role-specific shift comparison, accuracy summary |
| `requirements.txt` | Pinned dependencies (version-sensitive `spacy`/`opennyai`/`pydantic` stack) |
| `setup_workstation.sh` | Remote GPU workstation bootstrap script |
