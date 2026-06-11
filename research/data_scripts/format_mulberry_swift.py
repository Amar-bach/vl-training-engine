"""Convert Mulberry / VisionR1 cold-start traces into the Stage-B ms-swift SFT schema.

Goal: make Mulberry mixable with the Stage-B winner file in a single `swift sft` run, sharing
the SAME <think>/<answer> answer protocol so dataset weights are the only thing that differs.

Stage-B target schema (sft_stageB/train.jsonl):
    {"messages": [
        {"role": "system",    "content": "<protocol>"},
        {"role": "user",      "content": "<image>...question..."},
        {"role": "assistant", "content": "<grounding>...</grounding>\n<think>...</think>\n<answer>...</answer>"}],
     "images": ["/abs/path.webp"]}

Mulberry/VisionR1 has NO grounding block (no point annotations), so we emit only
<think>/<answer>. We keep the model's reasoning verbatim and normalize the answer wrapper to a
bare <answer> (dropping VisionR1's "Final Answer:" prefix) to match Stage-B exactly.

Image paths in the source are relative; we make them absolute against --img_base so swift can load
them. Records whose <think>/<answer> can't be recovered, or whose image is missing, are dropped
(counted in the report).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cot_lib'))
from cot_metrics import split_trace  # noqa: E402

# Light, format-only system prompt: same <think>/<answer> contract as Stage-B, minus grounding.
SYSTEM_PROMPT = (
    "You are a visual-reasoning assistant. Reason step by step inside a <think> block, "
    "then give a concise, definitive response inside an <answer> block."
)
_FINAL_ANS_RE = re.compile(r'^\s*(?:the\s+)?final\s+answer\s*[:\-]?\s*', re.I)
_IMG_TAG_RE = re.compile(r'<image>')


def _user_from(rec, schema):
    msgs = rec.get('conversations') if schema == 'conversations' else rec.get('messages')
    for m in (msgs or []):
        role = m.get('from') if schema == 'conversations' else m.get('role')
        if role in ('user', 'human'):
            return (m.get('value') if schema == 'conversations' else m.get('content')) or ''
    return ''


def _assistant_from(rec, schema):
    msgs = rec.get('conversations') if schema == 'conversations' else rec.get('messages')
    for m in reversed(msgs or []):
        role = m.get('from') if schema == 'conversations' else m.get('role')
        if role in ('assistant', 'gpt'):
            return (m.get('value') if schema == 'conversations' else m.get('content')) or ''
    return ''


def _images_field(rec):
    img = rec.get('images', rec.get('image'))
    if img is None:
        return []
    return img if isinstance(img, list) else [img]


def convert(rec, schema, img_base):
    user = _user_from(rec, schema)
    asst = _assistant_from(rec, schema)
    _, think, answer = split_trace(asst)
    if not think or not answer:
        return None, 'no_think_or_answer'
    answer = _FINAL_ANS_RE.sub('', answer).strip()
    if not answer:
        return None, 'empty_answer'

    rel_imgs = _images_field(rec)
    abs_imgs = []
    for p in rel_imgs:
        ap = p if Path(p).is_absolute() else str(Path(img_base) / p)
        if not Path(ap).exists():
            return None, 'image_missing'
        abs_imgs.append(ap)
    if not abs_imgs:
        return None, 'no_image'

    # ensure exactly one <image> tag in the user turn
    if not _IMG_TAG_RE.search(user):
        user = '<image>' + user
    new_asst = f"<think>{think}</think>\n<answer>{answer}</answer>"
    out = {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': new_asst},
        ],
        'images': abs_imgs,
    }
    return out, 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True)
    ap.add_argument('--schema', default='conversations', choices=['conversations', 'messages'])
    ap.add_argument('--img_base', required=True,
                    help='dir that relative image paths are resolved against')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='0 = all; else first N (for sampling)')
    args = ap.parse_args()

    data = json.load(open(args.path))
    if args.limit:
        data = data[:args.limit]
    from collections import Counter
    reasons = Counter()
    kept = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fo:
        for rec in data:
            out, why = convert(rec, args.schema, args.img_base)
            reasons[why] += 1
            if out is not None:
                fo.write(json.dumps(out, ensure_ascii=False) + '\n')
                kept += 1
    print(json.dumps({'in': len(data), 'kept': kept, 'reasons': dict(reasons),
                      'out': args.out}, indent=2))


if __name__ == '__main__':
    main()
