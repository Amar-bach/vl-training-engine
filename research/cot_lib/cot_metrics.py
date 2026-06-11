"""Shared CoT intrinsic-quality metrics.

These are the SAME trace-text features used in `audit_stage_b_winners.ipynb` (§9d/9f),
factored out so the Stage-B winner traces and the Mulberry/VisionR1 cold-start traces are
scored by *identical* definitions and can be compared apples-to-apples.

Judge-axis metrics (hallucination / visual_grounding / reasoning_quality / answer_correctness)
are NOT here — they require the Stage-B judge and do not exist for public SFT datasets.
What carries over is everything intrinsic to the reasoning TEXT itself.

A "trace" is the assistant turn. We split it into three optional spans:
    <grounding>...</grounding>   (Stage-B only; absent in Mulberry/VisionR1)
    <think>...</think>           (the reasoning we score)
    <answer>...</answer>         (the final answer)
If the tags are absent we fall back to treating the whole assistant text as `think`.
"""
import re
from collections import Counter

WORD_RE = re.compile(r"\S+")
OBJ_RE = re.compile(r"<obj\d+>")
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯ｦ-ﾟ]")
SENT_RE = re.compile(r"[.!?。！？]+(?:\s|$)")
CONNECT_RE = re.compile(
    r"\b(because|therefore|thus|hence|since|so that|as a result|which means|"
    r"this (?:means|implies|suggests|indicates|tells us)|consequently|given that)\b",
    re.IGNORECASE)
REFLECT_RE = re.compile(
    r"\b(wait|but actually|hmm|let me (?:check|reconsider|re-?evaluate|verify|think again)|"
    r"on second thought|reconsider|actually,|on closer|however|alternatively|"
    r"could (?:also|instead)|that would mean|let me re-?check)\b",
    re.IGNORECASE)

# Canonical self-improving-reasoner behaviors (identical to audit §9f).
COG_BEHAVIORS = {
    'backtracking': re.compile(
        r"\b(?:wait|hold on|scratch that|never ?mind|on second thought|back ?track(?:ing)?|"
        r"let me (?:redo|reconsider|rethink|re-?examine|re-?evaluate|start over|revise)|"
        r"that'?s (?:wrong|incorrect|not right|not correct)|i made a mistake|"
        r"that doesn'?t (?:work|seem right|match|add up)|but (?:that'?s|wait)|"
        r"going back|correction:|let me correct|hmm)\b", re.I),
    'verification': re.compile(
        r"\b(?:let me (?:verify|check|double-?check|confirm|make sure)|"
        r"to (?:verify|confirm|be sure|double-?check)|sanity[ -]?check|double check|"
        r"plug(?:ging)? (?:it |this |that )?back|does (?:this|that) (?:make sense|check out|hold|match)|"
        r"verify that|this (?:checks out|confirms|is consistent)|(?:is|are) consistent with|"
        r"confirm(?:s|ing)? (?:that|the)|as a check|let me confirm|to make sure)\b", re.I),
    'subgoal_setting': re.compile(
        r"\b(?:first,|firstly,|second,|secondly,|next,? (?:i|we|let|check|compare)|"
        r"then (?:i|we) (?:need|will|can|compare|check|look)|step \d|step one|step two|"
        r"let me (?:start|begin|first|determine|identify|establish|figure)|"
        r"i need to (?:first|determine|find|compare|figure|identify|establish)|"
        r"i'?ll (?:first|start by|need to)|break (?:this|it|the problem) (?:down|into)|"
        r"sub-?(?:goal|problem|task|step)|the (?:plan|steps?) (?:is|are)|"
        r"let'?s (?:start|begin|figure)|begin by|to (?:answer|solve) this,? (?:i|we|let))\b", re.I),
    'branching': re.compile(
        r"\b(?:alternativ(?:e|ely)|another (?:way|approach|option|possibility)|on the other hand|"
        r"case \d|case one|case two|in (?:one|the first|the second) case|option [ab12]\b|"
        r"two (?:possibilities|options|ways|cases)|we could (?:also|instead)|or (?:we|i) could|"
        r"consider (?:both|two|the other)|either way|but if|if (?:instead|however))\b", re.I),
    'backward_chaining': re.compile(
        r"\b(?:in order to|work(?:ing)? backwards?|(?:that|which) requires|"
        r"requires (?:knowing|finding|computing|first|the)|"
        r"to (?:determine|find|know|get|compute|answer)[^.,;]{1,55}?"
        r"(?:i need|we need|i must|we must|requires|need to know|need the|i first|we first)|"
        r"need to (?:know|find)[^.,;]{1,45}? (?:first|before)|so (?:i|we) first need|"
        r"depends on (?:knowing|the|whether))\b", re.I),
    'deduction': re.compile(
        r"\b(?:therefore|thus|hence|consequently|it follows that|as a result|"
        r"this (?:means|implies)|which (?:means|implies))\b", re.I),
}
COG_NAMES = list(COG_BEHAVIORS)

_FINAL_ANS_RE = re.compile(r'^\s*(?:the\s+)?final\s+answer\s*(?:is)?\s*[:\-]?\s*', re.I)
_GROUND_RE = re.compile(r"<grounding>(.*?)</grounding>", re.S | re.I)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_REASON_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.S | re.I)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def split_trace(assistant_text):
    """Return (grounding, think, answer) spans from an assistant turn.

    Handles <grounding>/<think>|<reasoning>/<answer>. If no <think>/<reasoning> tag is
    present, the whole text (minus grounding/answer) is treated as the reasoning span so
    untagged datasets still score.
    """
    s = assistant_text or ''
    g = _GROUND_RE.search(s)
    grounding = g.group(1).strip() if g else ''
    t = _THINK_RE.search(s) or _REASON_RE.search(s)
    a = _ANSWER_RE.search(s)
    answer = a.group(1).strip() if a else ''
    if t:
        think = t.group(1).strip()
    else:
        # no explicit reasoning tag: strip grounding+answer spans, keep the rest
        rest = _GROUND_RE.sub('', s)
        rest = _ANSWER_RE.sub('', rest)
        think = rest.strip()
    return grounding, think, answer


def _wc(s):
    return len(WORD_RE.findall(s or ''))


def cjk_frac(s):
    s = s or ''
    return len(CJK_RE.findall(s)) / max(1, len(s))


def max_ngram_rep(s, n=4):
    toks = WORD_RE.findall((s or '').lower())
    if len(toks) <= n:
        return 0.0
    grams = [' '.join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return Counter(grams).most_common(1)[0][1] / len(grams)


def ttr(s):
    toks = WORD_RE.findall((s or '').lower())
    return len(set(toks)) / max(1, len(toks))


def _norm(s):
    return re.sub(r'\s+', ' ', (s or '').strip().lower())


def trace_features(assistant_text):
    """All intrinsic + cognitive features for one assistant trace -> flat dict.

    Mirrors audit_stage_b_winners.ipynb §9d (intrinsic) + §9f (cognitive behaviors) so the
    columns line up exactly with the Stage-B reference.
    """
    grounding, think, answer = split_trace(assistant_text)
    defined = set(OBJ_RE.findall(grounding))
    used = defined & set(OBJ_RE.findall(think))
    # strip a leading "Final Answer:" wrapper (VisionR1 uses it; the SFT formatter drops it too)
    # so answer-derivation is checked on the answer BODY, consistent across datasets.
    a_norm = _norm(_FINAL_ANS_RE.sub('', answer))

    feats = dict(
        # --- lengths ---
        thinking_words=_wc(think),
        answer_words=_wc(answer),
        grounding_words=_wc(grounding),
        # --- reasoning structure ---
        sent_count=len(SENT_RE.findall(think)),
        connective_n=len(CONNECT_RE.findall(think)),
        # --- self-correction ---
        reflect_n=len(REFLECT_RE.findall(think)),
        has_reflection=bool(REFLECT_RE.search(think)),
        # --- language purity ---
        think_cjk_frac=cjk_frac(think),
        ans_cjk_frac=cjk_frac(answer),
        # --- degeneration / diversity ---
        rep_4gram=max_ngram_rep(think, 4),
        ttr=ttr(think),
        # --- grounding (Stage-B only; 0 for ungrounded datasets) ---
        obj_defined=len(defined),
        obj_used=len(used),
        ground_util=(len(used) / len(defined)) if defined else float('nan'),
        # --- answer derivation ---
        answer_in_think=bool(a_norm) and (a_norm in _norm(think)),
    )
    # cognitive behaviors (count + presence)
    n_present = 0
    for name, rx in COG_BEHAVIORS.items():
        c = len(rx.findall(think))
        feats[f'cog_{name}'] = c
        if c > 0:
            n_present += 1
    feats['cog_n_distinct'] = n_present

    # trace-level red flags (same thresholds as audit)
    feats['flag_cjk'] = feats['think_cjk_frac'] > 0.005
    feats['flag_degenerate'] = feats['rep_4gram'] > 0.10
    feats['flag_no_reflect'] = not feats['has_reflection']
    feats['flag_no_ground'] = (feats['obj_defined'] > 0) and (feats['obj_used'] == 0)
    feats['flag_ans_floating'] = not feats['answer_in_think']
    return feats


FLAG_LABELS = {
    'flag_cjk': 'CJK code-switch',
    'flag_degenerate': 'degenerate/looping',
    'flag_no_reflect': 'no self-correction',
    'flag_no_ground': 'grounding ignored',
    'flag_ans_floating': 'answer not derived in trace',
}
