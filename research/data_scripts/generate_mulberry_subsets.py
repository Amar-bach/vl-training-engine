"""Write per-domain-group subsets of the formatted Mulberry SFT file (notebook §9 logic).

Each subset is a filtered copy of mulberry_visionr1_train.jsonl containing only the records whose
source subtask (parsed from the image path) belongs to that domain group. Used as the Mulberry arm
of the SURDS+Mulberry ablation sweep.
"""
import json
import re
from collections import Counter
from pathlib import Path

R = Path('/mnt/data4/shasta/amar.amarjyoti/research_data')
MIX_DIR = R / 'sft_mulberry_visionr1'
SRC = MIX_DIR / 'mulberry_visionr1_train.jsonl'

DOMAIN = {
    'geometry_math':   {'geoqaplus', 'geo3k', 'unigeo', 'geos', 'mathvision',
                        'clevrmath', 'superclevr', 'tabmwp'},
    'chart_plot':      {'figureqa', 'plotqa', 'chartqa', 'dvqa', 'lrvchart', 'mapqa'},
    'science_diagram': {'ai2d', 'scienceqa', 'sqa', 'tqa', 'iconqa', 'pmcvqa'},
    'doc_text':        {'docvqa', 'infovqa', 'textvqa', 'vqaas'},
    'general_vqa':     {'cauldron', 'aokvqa', 'vqa20', 'vizwiz', 'vqarad'},
}
_canon2domain = {c: d for d, cs in DOMAIN.items() for c in cs}


def canon(name):
    return name.lower().replace('-', '').replace('.', '').replace('_images', '').replace('_', '')


def main():
    writers = {d: open(MIX_DIR / f'mulberry_{d}_train.jsonl', 'w') for d in DOMAIN}
    counts = Counter()
    other = Counter()
    n = 0
    with open(SRC) as f:
        for line in f:
            n += 1
            m = re.search(r'mulberry_images/([^/]+)/', line)
            c = canon(m.group(1)) if m else '(unknown)'
            d = _canon2domain.get(c)
            if d:
                writers[d].write(line)
                counts[d] += 1
            else:
                other[c] += 1
    for w in writers.values():
        w.close()
    print(f'scanned {n:,} records from {SRC.name}\n')
    for d in DOMAIN:
        p = MIX_DIR / f'mulberry_{d}_train.jsonl'
        print(f'  {d:16s} {counts[d]:8,}  -> {p}')
    if other:
        print('\n  uncategorized (NOT written):', dict(other))
    # machine-readable summary for the job launcher
    (MIX_DIR / 'subset_counts.json').write_text(json.dumps(
        {d: counts[d] for d in DOMAIN} | {'full': n}, indent=2))
    print('\nwrote', MIX_DIR / 'subset_counts.json')


if __name__ == '__main__':
    main()
