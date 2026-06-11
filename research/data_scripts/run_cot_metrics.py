"""Compute intrinsic CoT metrics over one dataset and dump a per-record parquet + summary.

Reads either:
  * a JSONL file with ms-swift schema  ({"messages":[...], "images": ...})        -> --schema messages
  * a JSON array (Mulberry / VisionR1)  ({"conversations":[{from,value}], ...})    -> --schema conversations
  * a JSON array original Mulberry      ({"messages":[{role,content}], ...})       -> --schema messages_json

Usage:
  python run_cot_metrics.py --name stage_b   --path .../sft_stageB/train.jsonl              --schema messages
  python run_cot_metrics.py --name mulberry  --path .../vision_r1_mulberry_sft_full.json     --schema conversations
  python run_cot_metrics.py --name llava_cot --path .../vision_r1_llava_cot_full.json         --schema conversations
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cot_lib'))
from cot_metrics import trace_features, COG_NAMES, FLAG_LABELS  # noqa: E402


def _assistant_from_messages(rec):
    for m in reversed(rec.get('messages', [])):
        if m.get('role') == 'assistant':
            return m.get('content', '')
    return ''


def _assistant_from_conversations(rec):
    for m in reversed(rec.get('conversations', [])):
        if m.get('from') in ('assistant', 'gpt'):
            return m.get('value', '')
    return ''


def iter_records(path, schema):
    path = Path(path)
    if schema == 'messages' and path.suffix == '.jsonl':
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    yield json.loads(ln)
    else:
        # JSON array (possibly large) — load once
        data = json.load(open(path))
        for rec in data:
            yield rec


def extract_assistant(rec, schema):
    if schema == 'conversations':
        return _assistant_from_conversations(rec)
    return _assistant_from_messages(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--path', required=True)
    ap.add_argument('--schema', required=True,
                    choices=['messages', 'conversations', 'messages_json'])
    ap.add_argument('--out_dir', default=str(Path(__file__).resolve().parents[2] / 'notebooks' / 'visionr1_out'))
    args = ap.parse_args()

    schema = 'messages' if args.schema == 'messages_json' else args.schema
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    t0 = time.time()
    rows = []
    n = 0
    for rec in iter_records(args.path, schema):
        asst = extract_assistant(rec, schema)
        rows.append(trace_features(asst))
        n += 1
        if n % 25000 == 0:
            print(f'[{args.name}] {n:,} records  ({time.time()-t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df['dataset'] = args.name

    pq = out_dir / f'{args.name}_metrics.parquet'
    df.to_parquet(pq, index=False)

    # ---- summary ----
    num_cols = ['thinking_words', 'answer_words', 'grounding_words', 'sent_count',
                'connective_n', 'reflect_n', 'think_cjk_frac', 'rep_4gram', 'ttr',
                'ground_util', 'cog_n_distinct'] + [f'cog_{c}' for c in COG_NAMES]
    summary = {
        'name': args.name,
        'path': args.path,
        'n_records': int(n),
        'means': {c: float(df[c].mean()) for c in num_cols if c in df},
        'medians': {c: float(df[c].median()) for c in num_cols if c in df},
        'quantiles_thinking_words': {q: float(df.thinking_words.quantile(q))
                                     for q in [0.5, 0.9, 0.95, 0.99, 1.0]},
        'rate_has_reflection': float(df.has_reflection.mean()),
        'rate_answer_in_think': float(df.answer_in_think.mean()),
        'rate_any_cjk_think': float((df.think_cjk_frac > 0).mean()),
        'cog_present_rate': {c: float((df[f'cog_{c}'] > 0).mean()) for c in COG_NAMES},
        'flag_rate': {FLAG_LABELS[f]: float(df[f].mean()) for f in FLAG_LABELS},
        'rate_ge3_behaviors': float((df.cog_n_distinct >= 3).mean()),
        'rate_0_behaviors': float((df.cog_n_distinct == 0).mean()),
        'elapsed_s': round(time.time() - t0, 1),
    }
    (out_dir / f'{args.name}_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[{args.name}] DONE  {n:,} records -> {pq.name}  ({summary["elapsed_s"]}s)', flush=True)


if __name__ == '__main__':
    main()
