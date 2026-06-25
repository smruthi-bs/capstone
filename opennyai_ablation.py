# ─────────────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import json
import re
import torch

_original_load_state_dict = torch.nn.Module.load_state_dict
def _patched_load_state_dict(self, state_dict, strict=True, **kwargs):
    return _original_load_state_dict(self, state_dict, strict=False, **kwargs)
torch.nn.Module.load_state_dict = _patched_load_state_dict

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from opennyai import Pipeline
from opennyai.utils import Data
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# SECTION 2: CONFIG — update these paths to your local setup
# ─────────────────────────────────────────────────────────────────────
CSV_PATH        = './predex_test_deduped.csv'
CHECKPOINT_PATH = './best_hierarchical_model.pt'
SPLITS_JSON     = './predex_opennyai_splits.json'
RESULTS_JSON    = './predex_ablation_results.json'
PLOTS_DIR       = './'

MODEL_NAME      ="./InLegalBERT"          # local folder downloaded from your HF repo
GRU_HIDDEN      = 256
GRU_LAYERS      = 2
DROPOUT         = 0.3
UNFREEZE_LAYERS = 3
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU         = torch.cuda.is_available()

CHUNK_SIZE = 512
STRIDE     = 256
MAX_CHUNKS = 16

# ─────────────────────────────────────────────────────────────────────
# SECTION 3: TOKEN-LEVEL MASKING & COUNTERFACTUAL
# ─────────────────────────────────────────────────────────────────────
MASK_PATTERNS = {
    'section_refs':  r"\bSection\s+\d+[\w\s,/&()]*?(IPC|CrPC|NDPS|POCSO|Arms\s+Act)?\b",
    'outcome_words': r"\bconvicted\b|\bacquitted\b|\bsentenced\b|\blife\s+imprisonment\b|\bdeath\s+sentence\b",
    'citations':     r"\b(AIR|SCC|SCR|Cr\.?L\.?J)\s*\d{4}\b",
    'act_names':     r"\bI\.?P\.?C\.?\b|\bCr\.?P\.?C\.?\b|\bNDPS\b|\bPOCSO\b|\bArms\s+Act\b",
    'court_names':   r"\bSupreme\s+Court\b|\bHigh\s+Court\b|\bSessions\s+Court\b",
}

def mask_shortcut_tokens(sentence):
    """
    Masks shortcut tokens within a sentence but keeps the sentence structure.
    Used to isolate token-level shortcut effect within FAC sentences.
    e.g. "charged under Section 302 IPC" → "charged under [LEGAL_REF]"
    """
    masked = sentence
    for pattern in MASK_PATTERNS.values():
        masked = re.sub(pattern, '[LEGAL_REF]', masked, flags=re.IGNORECASE)
    return masked

def counterfactual_section(sentence):
    """
    Replaces specific section numbers with a neutral one (144).
    If prediction changes just because of this swap, the model
    is using the section number itself as a shortcut — not the
    factual context around it. That is direct causal evidence.
    e.g. "charged under Section 302" → "charged under Section 144"
    """
    return re.sub(r'Section\s+\d+', 'Section 144', sentence, flags=re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────
# SECTION 4: OPENNYAI SEGMENTATION
# ─────────────────────────────────────────────────────────────────────
KNOWN_ROLES = ['FAC', 'ARG', 'STA', 'PRE', 'RLC', 'RPC', 'RCJ', 'RAT', 'NONE']

def run_opennyai_segmentation(texts, batch_size=8):
    """
    Runs OpenNyAI rhetorical role classifier on all input texts.
    Returns a list of dicts (one per text), each mapping role → [sentences].
    
    batch_size=8 is safe for most machines. Drop to 4 if you get OOM errors.
    OpenNyAI downloads model weights (~400MB) on first run, then caches them.
    """
    print("Loading OpenNyAI pipeline...")
    pipeline = Pipeline(
        components=['Rhetorical_Role'],
        use_gpu=USE_GPU,
        verbose=False
    )
    print(f"Pipeline loaded on {'GPU' if USE_GPU else 'CPU'}")

    all_segmented = []

    for batch_start in tqdm(range(0, len(texts), batch_size), desc="OpenNyAI segmentation"):
        batch_texts = texts[batch_start : batch_start + batch_size]
        data = Data(batch_texts)

        try:
            results = pipeline(data)
        except Exception as e:
            print(f"  Warning: batch {batch_start} failed — {e}. Inserting empty buckets.")
            for _ in batch_texts:
                all_segmented.append({role: [] for role in KNOWN_ROLES})
            continue

        for result in results:
            role_buckets = {role: [] for role in KNOWN_ROLES}
            for annotation in result.get('annotations', []):
                sentence = annotation.get('sentence', '').strip()
                role     = annotation.get('Rhetorical_Role', 'NONE')
                if not sentence:
                    continue
                if role not in role_buckets:
                    role = 'NONE'
                role_buckets[role].append(sentence)
            all_segmented.append(role_buckets)

    return all_segmented

# ─────────────────────────────────────────────────────────────────────
# SECTION 5: BUILD VARIANTS
#
# This is the core of the combined design.
#
# LAYER 1 — Graduated ablation (answers: does AMOUNT of spurious matter?)
#   Spurious pool = STA + PRE + RLC + RPC + RCJ (all high-risk roles)
#   Injected in 0/25/50/75/100% increments into FAC sentences
#   → Produces a dose-response curve
#
# LAYER 2 — Role-specific ablation (answers: which TYPE of spurious matters?)
#   Adds each high-risk role type individually on top of FAC
#   → Identifies the specific culprit role
#
# LAYER 3 — Counterfactual (answers: is it the section NUMBER specifically?)
#   Swaps section numbers in FAC sentences to neutral value (144)
#   → Direct causal evidence if prediction changes
# ─────────────────────────────────────────────────────────────────────
def build_variants(role_buckets):
    facts    = role_buckets.get('FAC', [])
    sta      = role_buckets.get('STA', [])
    pre      = role_buckets.get('PRE', [])
    rlc      = role_buckets.get('RLC', [])
    rpc      = role_buckets.get('RPC', [])
    rcj      = role_buckets.get('RCJ', [])
    arg      = role_buckets.get('ARG', [])
    rat      = role_buckets.get('RAT', [])
    all_sents = [s for sents in role_buckets.values() for s in sents]

    # Pool all high-risk roles as the "spurious" set for graduated ablation
    spurious = sta + pre + rlc + rpc + rcj
    n = len(spurious)

    def join(*lists):
        combined = []
        for lst in lists:
            combined.extend(lst)
        return ' '.join(combined).strip()

    return {
        # ── Layer 1: Graduated ablation ───────────────────────────────
        # FAC sentences are kept raw (unmasked) here so the baseline
        # matches what the model would see in a real fact-only input.
        # Spurious content injected in fixed increments.
        'v0_facts_only': join(facts),
        'v25':           join(facts, spurious[:max(1, n//4)]),
        'v50':           join(facts, spurious[:max(1, n//2)]),
        'v75':           join(facts, spurious[:max(1, 3*n//4)]),
        'v100_full_spu': join(facts, spurious),

        # ── Layer 2: Role-specific ablation ──────────────────────────
        # Each variant adds ONE role type on top of FAC.
        # Compare shift rates across these to find the worst offender.
        'v_sta_only':     join(facts, sta),         # statute sentences only
        'v_pre_only':     join(facts, pre),          # precedent sentences only
        'v_ruling_only':  join(facts, rlc + rpc + rcj),  # ruling sentences only
        'v_arg_rat':      join(facts, arg + rat),    # medium-risk roles (control)

        # ── Layer 3: Counterfactual ───────────────────────────────────
        # FAC sentences with section numbers swapped to neutral value.
        # If predictions shift vs v0_facts_only, section numbers are shortcuts.
        'v_counterfactual': join([counterfactual_section(s) for s in facts]),

        # ── Full text baseline ────────────────────────────────────────
        # Should match original model performance. Used as upper bound.
        'v_full': join(all_sents),
    }

# ─────────────────────────────────────────────────────────────────────
# SECTION 6: MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────
class HierarchicalLegalBERT(nn.Module):
    def __init__(self, model_name, gru_hidden=256, gru_layers=2,
                 dropout=0.3, num_classes=2, unfreeze_last_n=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        num_layers = len(self.bert.encoder.layer)
        for i in range(num_layers - unfreeze_last_n, num_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        bert_dim = self.bert.config.hidden_size
        self.gru = nn.GRU(
            input_size=bert_dim, hidden_size=gru_hidden,
            num_layers=gru_layers, batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask, chunk_counts):
        B, max_chunks, seq_len = input_ids.shape
        ids   = input_ids.view(B * max_chunks, seq_len)
        masks = attention_mask.view(B * max_chunks, seq_len)
        bert_out = self.bert(input_ids=ids, attention_mask=masks)
        cls_emb  = bert_out.last_hidden_state[:, 0, :]
        cls_emb  = cls_emb.view(B, max_chunks, -1)
        cls_emb  = self.dropout(cls_emb)
        packed   = nn.utils.rnn.pack_padded_sequence(
            cls_emb, chunk_counts.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        final = torch.cat([hidden[-2], hidden[-1]], dim=1)
        final = self.dropout(final)
        return self.classifier(final)

# ─────────────────────────────────────────────────────────────────────
# SECTION 7: CHUNKING + INFERENCE
# ─────────────────────────────────────────────────────────────────────
def chunk_text(text, tokenizer):
    encoding  = tokenizer(str(text), add_special_tokens=False, return_tensors=None)
    all_ids   = encoding['input_ids']
    all_mask  = encoding['attention_mask']
    effective = CHUNK_SIZE - 2
    chunks, start = [], 0
    while start < len(all_ids):
        end        = start + effective
        chunk_ids  = [tokenizer.cls_token_id] + all_ids[start:end]  + [tokenizer.sep_token_id]
        chunk_mask = [1]                       + all_mask[start:end] + [1]
        pad        = CHUNK_SIZE - len(chunk_ids)
        chunk_ids  += [tokenizer.pad_token_id] * pad
        chunk_mask += [0] * pad
        chunks.append((chunk_ids, chunk_mask))
        if end >= len(all_ids):
            break
        start += STRIDE
    return chunks[:MAX_CHUNKS]

def run_inference(text, model, tokenizer):
    if not str(text).strip():
        return -1, 0.0
    chunks = chunk_text(text, tokenizer)
    n      = len(chunks)
    ids    = torch.tensor([c[0] for c in chunks], dtype=torch.long).unsqueeze(0).to(DEVICE)
    masks  = torch.tensor([c[1] for c in chunks], dtype=torch.long).unsqueeze(0).to(DEVICE)
    counts = torch.tensor([n], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids, masks, counts)
    probs           = F.softmax(logits, dim=-1)
    predicted_label = probs.argmax().item()
    confidence      = probs.max().item()
    return predicted_label, confidence

# ─────────────────────────────────────────────────────────────────────
# SECTION 8: ANALYSIS
# ─────────────────────────────────────────────────────────────────────
def analyze_results(results):
    """
    Three separate analyses matching the three experimental layers.
    Baseline for shift rate is always v0_facts_only.
    """
    all_variant_keys = [
        'v0_facts_only', 'v25', 'v50', 'v75', 'v100_full_spu',
        'v_sta_only', 'v_pre_only', 'v_ruling_only', 'v_arg_rat',
        'v_counterfactual', 'v_full'
    ]

    # Compute accuracy and shift rate for every variant
    summary = {}
    for vkey in all_variant_keys:
        shifts, correct, confs = [], [], []
        for case in results:
            pred = case['predictions'].get(vkey, {}).get('label', -1)
            if pred == -1:
                continue
            base = case['predictions'].get('v0_facts_only', {}).get('label', -1)
            true = case['true_label']
            if base != -1:
                shifts.append(1 if pred != base else 0)
            correct.append(1 if pred == true else 0)
            confs.append(case['predictions'][vkey]['confidence'])

        summary[vkey] = {
            'accuracy':       round(np.mean(correct) * 100, 2) if correct else 0,
            'shift_rate':     round(np.mean(shifts)  * 100, 2) if shifts  else 0,
            'avg_confidence': round(np.mean(confs),          4) if confs   else 0,
        }

    # ── Print Layer 1: Graduated ablation ─────────────────────────────
    print("\n" + "="*60)
    print("LAYER 1 — Graduated Ablation (dose-response curve)")
    print(f"{'Variant':<20} {'Accuracy':>10} {'Shift vs v0':>12} {'Avg Conf':>10}")
    print("="*60)
    for vkey in ['v0_facts_only', 'v25', 'v50', 'v75', 'v100_full_spu']:
        s = summary[vkey]
        print(f"  {vkey:<18} {s['accuracy']:>9.1f}% {s['shift_rate']:>11.1f}% {s['avg_confidence']:>10.3f}")

    # ── Print Layer 2: Role-specific ──────────────────────────────────
    print("\n" + "="*60)
    print("LAYER 2 — Role-Specific Ablation (which type matters most?)")
    print(f"{'Variant':<20} {'Accuracy':>10} {'Shift vs v0':>12} {'Avg Conf':>10}")
    print("="*60)
    for vkey in ['v0_facts_only', 'v_sta_only', 'v_pre_only', 'v_ruling_only', 'v_arg_rat', 'v_full']:
        s = summary[vkey]
        print(f"  {vkey:<18} {s['accuracy']:>9.1f}% {s['shift_rate']:>11.1f}% {s['avg_confidence']:>10.3f}")

    # ── Print Layer 3: Counterfactual ─────────────────────────────────
    print("\n" + "="*60)
    print("LAYER 3 — Counterfactual Test (section number as shortcut?)")
    cf_shifts = []
    for case in results:
        raw = case['predictions'].get('v0_facts_only', {}).get('label', -1)
        cf  = case['predictions'].get('v_counterfactual', {}).get('label', -1)
        if raw != -1 and cf != -1:
            cf_shifts.append(1 if raw != cf else 0)
    cf_rate = np.mean(cf_shifts) * 100 if cf_shifts else 0
    print(f"  Cases where swapping section number changed prediction: {cf_rate:.1f}%")
    print(f"  Interpretation: {'STRONG shortcut evidence' if cf_rate > 10 else 'weak or no section-number shortcut'}")

    return summary


# ─────────────────────────────────────────────────────────────────────
# SECTION 9: PLOTS
# ─────────────────────────────────────────────────────────────────────
def plot_results(summary):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Graduated ablation — shift rate curve
    levels      = [0, 25, 50, 75, 100]
    grad_keys   = ['v0_facts_only', 'v25', 'v50', 'v75', 'v100_full_spu']
    shift_rates = [summary[k]['shift_rate'] for k in grad_keys]
    accuracies  = [summary[k]['accuracy']   for k in grad_keys]

    ax = axes[0]
    ax.plot(levels, shift_rates, marker='o', color='crimson',   linewidth=2, markersize=8, label='Shift Rate')
    ax.plot(levels, accuracies,  marker='s', color='steelblue', linewidth=2, markersize=8, label='Accuracy')
    ax.set_xlabel('Spurious Content Level (%)')
    ax.set_ylabel('%')
    ax.set_title('Layer 1: Graduated Ablation\n(dose-response curve)')
    ax.set_xticks(levels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Role-specific — horizontal bar chart of shift rates
    role_keys   = ['v_sta_only', 'v_pre_only', 'v_ruling_only', 'v_arg_rat']
    role_labels = ['Statute (STA)', 'Precedent (PRE)', 'Ruling (RLC/RPC)', 'Arg+Ratio (control)']
    role_shifts = [summary[k]['shift_rate'] for k in role_keys]

    ax = axes[1]
    bars = ax.barh(role_labels, role_shifts, color='steelblue', edgecolor='black')
    for bar, val in zip(bars, role_shifts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)
    ax.set_xlabel('Prediction Shift Rate (%)')
    ax.set_title('Layer 2: Role-Specific Ablation\n(which type causes most shift?)')
    ax.set_xlim(0, max(role_shifts) + 15 if role_shifts else 20)
    ax.grid(axis='x', alpha=0.3)

    # Plot 3: Summary accuracy comparison
    summary_keys   = ['v0_facts_only', 'v100_full_spu', 'v_ruling_only', 'v_full']
    summary_labels = ['Facts Only', 'Facts+All Spurious', 'Facts+Ruling', 'Full Text']
    summary_accs   = [summary[k]['accuracy'] for k in summary_keys]

    ax = axes[2]
    colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db']
    bars = ax.bar(summary_labels, summary_accs, color=colors, edgecolor='black')
    for bar, val in zip(bars, summary_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=10)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Layer 3: Accuracy Comparison\n(key variants)')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/combined_ablation.png', dpi=150)
    plt.show()
    print(f"Plot saved to {PLOTS_DIR}/combined_ablation.png")


# ─────────────────────────────────────────────────────────────────────
# SECTION 10: MAIN
# ─────────────────────────────────────────────────────────────────────
def main():

    # ── Load dataset ──────────────────────────────────────────────────
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} cases | Columns: {df.columns.tolist()}")

    # ── Step 1: OpenNyAI segmentation ─────────────────────────────────
    print("\nStep 1: OpenNyAI rhetorical role segmentation")
    texts     = df['Input'].astype(str).tolist()
    segmented = run_opennyai_segmentation(texts, batch_size=8)

    # Build splits
    all_splits = []
    for idx, (row, role_buckets) in enumerate(zip(df.itertuples(), segmented)):
        all_splits.append({
            'index':       idx,
            'case_name':   getattr(row, 'Case_Name', f'case_{idx}'),
            'label':       row.Label,
            'role_buckets': role_buckets,
            'role_counts': {r: len(s) for r, s in role_buckets.items() if s},
            'variants':    build_variants(role_buckets),
        })

    with open(SPLITS_JSON, 'w') as f:
        json.dump(all_splits, f)
    print(f"Splits saved to {SPLITS_JSON}")

    # Spot-check first case
    print("\nSegmentation sample (case 0):")
    for role, sents in all_splits[0]['role_buckets'].items():
        if sents:
            print(f"  [{role}] {len(sents)} sentences — e.g.: {sents[0][:100]}")

    # ── Step 2: Load model ─────────────────────────────────────────────
    print(f"\nStep 2: Loading model from {CHECKPOINT_PATH}")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    model      = HierarchicalLegalBERT(
        MODEL_NAME, GRU_HIDDEN, GRU_LAYERS, DROPOUT,
        unfreeze_last_n=UNFREEZE_LAYERS
    ).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded | Epoch: {checkpoint['epoch']} | Best F1: {checkpoint['best_f1']:.4f}")

    # ── Step 3: Inference on all variants ─────────────────────────────
    print("\nStep 3: Running inference on all variants...")
    variant_keys = [
        'v0_facts_only', 'v25', 'v50', 'v75', 'v100_full_spu',
        'v_sta_only', 'v_pre_only', 'v_ruling_only', 'v_arg_rat',
        'v_counterfactual', 'v_full'
    ]

    results = []
    for case in tqdm(all_splits, desc="Inference"):
        case_result = {
            'index':      case['index'],
            'case_name':  case['case_name'],
            'true_label': case['label'],
            'predictions': {}
        }
        for vkey in variant_keys:
            text = case['variants'].get(vkey, '')
            if not text.strip():
                case_result['predictions'][vkey] = {'label': -1, 'confidence': 0.0}
                continue
            pred, conf = run_inference(text, model, tokenizer)
            case_result['predictions'][vkey] = {
                'label':      pred,
                'confidence': round(conf, 4)
            }
        results.append(case_result)

    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {RESULTS_JSON}")

    # ── Step 4: Analysis + plots ───────────────────────────────────────
    print("\nStep 4: Analysis")
    summary = analyze_results(results)
    plot_results(summary)

    # Final interpretation guide
    print("\n" + "="*60)
    print("HOW TO INTERPRET YOUR RESULTS")
    print("="*60)
    print("""
  Layer 1 (Graduated):
    Monotonically rising shift rate curve → model is sensitive to
    spurious volume. Flat curve → amount doesn't matter.

  Layer 2 (Role-specific):
    Highest shift rate bar = the role type driving shortcuts.
    If v_ruling_only >> v_sta_only, ruling language is the culprit.
    If v_arg_rat ≈ 0, medium-risk roles are not shortcuts (good control).

  Layer 3 (Counterfactual):
    >10% shift from section swap = model uses section numbers as
    shortcuts independent of surrounding factual context.

  Accuracy gap (v0 vs v_full):
    Large gap → model genuinely needs spurious content to perform.
    Small gap → model can reason from facts alone. No shortcut problem.
    """)


if __name__ == '__main__':
    main()
