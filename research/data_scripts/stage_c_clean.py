"""Deterministic, answer-preserving cleanup of "description-as-text" deformities in
DeepSeek-generated CoT traces (Stage-C) for the VL SFT dataset.

These traces were produced text-only from a *scene description*, so they refer to
"the description", "we are given a description of an image", "the text says", "is
described as", etc. The student model actually SEES the image, so we rewrite that
text-reading framing into image-grounded phrasing WITHOUT touching any conclusion,
object, number, or coordinate.

Public API:
    clean_think(text)  -> str
    clean_answer(text) -> str
    DEFORMITY_PATTERNS -> list[compiled regex]   (used by the audit)

Design notes:
- Substitutions are ORDERED: specific/long patterns run before generic ones.
- `task description` / `the question` / a quoted sign like the text "INDOCUNA" refer
  to the *task* or to literal image text, NOT to the scene description, and are left
  untouched on purpose.
- No rule ever rewrites digits, bracketed coordinates, or decisions.
- Whitespace / capitalization / dangling colons are normalized at the very end.
"""

import re

# A "description verb" head: the words that follow "the description ..." when it is
# being used as a text source. Used to detect deformities and to gate "the text".
_DESC_VERB = (
    r"(?:explicitly\s+|clearly\s+|also\s+|only\s+|again\s+|repeatedly\s+|carefully\s+)*"
    r"(?:says|states|state|said|mentions|mention|mentioned|notes|noted|indicates|"
    r"indicate|tells\s+us|makes\s+clear|clarifies|confirms|confirm|defines|define|"
    r"provides|provide|gives|give|specifies|specify|describes|describe|reads|read|"
    r"lists|list|shows|show|reports|report|suggests|suggest)"
)

# ---------------------------------------------------------------------------
# Ordered (pattern, replacement) rules. Each pattern is compiled case-insensitive.
# Replacements use \g<...> backrefs to preserve captured connectives/verbs/objects.
# IMPORTANT: ordering matters — earlier rules are more specific.
# ---------------------------------------------------------------------------

# Negative lookbehind to never touch "task description".
_NOT_TASK = r"(?<!task )(?<!Task )"

_RULES_RAW = [
    # --- 0/1. Opening "we are given a (detailed) description of <FRAME>" framing --
    # CRITICAL: these rules replace ONLY the framing noun-phrase, never any scene
    # content that follows it (objects, coordinates). The framing tail can be:
    #   - "an/the image" / "a/the scene" / "a <adj> scene image"
    #   - "the spatial relationships/layout/positions/relative sizes/distances ..."
    #     between/of/in <objects>  (we keep the objects)
    # We match the framing head + a bounded, content-free connective ("of the
    # spatial relationships between", etc.) and stop BEFORE any object words.

    # 0a. framing followed by a colon + quote/content: "..description: \"X\"" ->
    #     keep the content, just reframe.  (no swallowing of the quote)
    (r"\bwe are given a (?:detailed )?description of "
     r"(?:an?|the) (?:driving |traffic )?(?:image|scene)(?: image)?"
     r"\s*:\s*(?=\")",
     "Looking at this image, we can see: "),
    (r"\bwe are given a (?:detailed )?description\s*:\s*(?=\")",
     "Looking at this image, we can see: "),

    # 0b. framing "of <FRAME> [between|of|in|with|that|providing] ..." — replace only
    #     up to and including the connective preposition, KEEP following objects.
    (r"\bwe are given a (?:detailed )?description of "
     r"(?:the |a )?(?:spatial (?:relationships?|layout|configuration|arrangement|relations?)"
     r"|(?:positions?|locations?|placement)(?: and relative (?:sizes?|distances?|positions?))?"
     r"|relative (?:positions?|sizes?|distances?|locations?))"
     r"\s+(?:between|of|in|among|for)\s+",
     "Looking at this image, considering "),

    # 0c. framing that ends a sentence: "...description of <FRAME>. " -> sentence form
    (r"\bwe are given a (?:detailed )?description of "
     r"(?:an?|the) (?:driving |traffic )?(?:image|scene)(?: image)?\b\s*\.\s*",
     "Looking at this image, "),
    (r"\bwe are given a (?:detailed )?description of "
     r"(?:the |a )?(?:spatial (?:relationships?|layout|configuration|arrangement|relations?)"
     r"|(?:positions?|locations?|placement)(?: and relative (?:sizes?|distances?|positions?))?"
     r"|relative (?:positions?|sizes?|distances?|locations?))\s*\.\s*",
     "Looking at this image, "),

    # 0d. framing followed by "with/that/featuring/showing/which ..." (keeps the rest)
    (r"\bwe are given a (?:detailed )?description of "
     r"(?:an?|the) (?:driving |traffic )?(?:image|scene)(?: image)?\b\s*"
     r"(?=with\b|that\b|featuring\b|showing\b|which\b|containing\b|including\b)",
     "Looking at this image "),
    (r"\bwe are given a (?:detailed )?description of (?:an?|the) (?:image|scene)\b\s*,\s*",
     "Looking at this image, "),
    (r"\bwe are given a (?:detailed )?description\b\s*(?=with\b|that\b|which\b)",
     "Looking at this image "),

    # 0e-1. "description of two/three objects|vehicles : <content>" — keep the
    #       enumerated objects; only reframe the lead-in (stop AT the colon).
    (r"\bwe are given a (?:detailed )?description of "
     r"(two|three|four|several|multiple|the|these) (objects?|vehicles?|items?)\s*:\s*",
     r"Looking at this image, we can see \g<1> \g<2>: "),
    (r"\bwe are given a (?:detailed )?description of "
     r"(two|three|four|several|multiple) (objects?|vehicles?|items?)\b\s*"
     r"(?=,|\.|\bin\b|\bwith\b)",
     r"Looking at this image, we can see \g<1> \g<2> "),
    # 0e-2. "description of a/an <adjectives> scene . The" -> sentence reframe
    #       (period boundary; never crosses it).
    (r"\bwe are given a (?:detailed )?description of (?:a|an) [\w\- ]{0,40}?scene\b\s*\.\s*",
     "Looking at this image, "),
    # 0e-3. "description: <content>" (no leading quote — handled by 0a) -> reframe
    #       lead-in only, keep content after the colon.
    (r"\bwe are given a (?:detailed )?description\s*:\s*",
     "Looking at this image, we can see: "),
    # 0e-4. "description (and coordinates|from a vehicle's perspective|of objects in
    #        a scene| and need ...)" -> reframe lead-in up to a safe boundary.
    (r"\bwe are given a (?:detailed )?description\b\s*"
     r"(?=and\b|from\b|of (?:two|three|a|an|the|objects?|vehicles?)\b)",
     "Looking at this image, we have a view "),
    # 0e-5. fallback: any remaining "we are given a description of an image/scene"
    #       not caught above (no clear boundary) -> sentence reframe.
    (r"\bwe are given a (?:detailed )?description of (?:an?|the) (?:image|scene)\b\s*\.?\s*",
     "Looking at this image. "),

    # "Given a (detailed) description of the image/scene/spatial relationships. The"
    (r"\bgiven a (?:detailed )?description of (?:an?|the) (?:image|scene)\b\s*\.\s*",
     "Looking at this image, "),
    (r"\bgiven a (?:detailed )?description of "
     r"(?:the |a )?(?:spatial (?:relationships?|layout|configuration|arrangement|relations?)"
     r"|(?:positions?|locations?|placement)(?: and relative (?:sizes?|distances?|positions?))?"
     r"|relative (?:positions?|sizes?|distances?|locations?))"
     r"\s+(?:between|of|in|among|for)\s+",
     "Looking at this image, considering "),
    (r"\bgiven a (?:detailed )?description of "
     r"(?:the |a )?(?:spatial (?:relationships?|layout|configuration|arrangement|relations?)"
     r"|(?:positions?|locations?|placement)(?: and relative (?:sizes?|distances?|positions?))?"
     r"|relative (?:positions?|sizes?|distances?|locations?))\s*\.\s*",
     "Looking at this image, "),

    # --- 2. Connective + description --------------------------------------------
    # "from/based on/according to (the/this) (scene) description" -> "... the image"
    (r"\b(from|based on|according to|using|per)\s+(?:this|the|a)\s+(?:scene |detailed )?description\b" + r"(?![\w])",
     r"\g<1> the image"),
    (r"\bfrom (?:the )?description\s*,",
     "from the image,"),
    (r"\bfrom (?:the )?description\s*:",
     "from the image:"),
    # "in this/the description" -> "in the image"
    (r"\bin (?:this|the) description\b", "in the image"),
    # "mentioned in (this/the) (scene) description" -> "visible in the image"
    (r"\bmentioned in (?:this|the) (?:scene )?description\b", "visible in the image"),

    # --- 3. "the (scene) description <verb>" -> "the image <verb>" ---------------
    # Do NOT touch "task description". Keep the verb (mapped to a natural image verb
    # only where the description verb is itself a reading verb).
    (_NOT_TASK + r"\bthe (?:scene )?description\b(\s+" + _DESC_VERB + r")",
     r"the image\g<1>"),
    # "the description provides/gives coordinates|spatial ..." already covered above
    # via _DESC_VERB (provides/gives). Bare "the (scene) description" -> "the image".
    (_NOT_TASK + r"\bthe scene description\b", "the image"),
    (_NOT_TASK + r"\bthe given description\b", "the image"),
    (_NOT_TASK + r"\bthis description\b", "the image"),
    (_NOT_TASK + r"\bthe description\b", "the image"),

    # --- 4. "is/are described as" -> "appears / appear" -------------------------
    # Special-case "described as being" -> "appears to be" for grammaticality.
    (r"\bis described as being\b", "appears to be"),
    (r"\bare described as being\b", "appear to be"),
    (r"\bis described as\b", "appears"),
    (r"\bare described as\b", "appear"),
    (r"\bis described\b", "is shown"),
    (r"\bare described\b", "are shown"),
    (r"\bas described\b", "as shown"),

    # --- 5. "the text <desc-verb>" -> "the image <verb>" ------------------------
    # Only when 'the text' is used as a reading source (followed by a desc verb),
    # NOT e.g. the text "INDOCUNA" (a literal sign in the image).
    (r"\bthe text\b(\s+" + _DESC_VERB + r")", r"the image\g<1>"),
    (r"\bthe text\s*:", "the image:"),
    (r"\bpart of the text\b", "part of the image"),
    (r"\bthe passage\b", "the image"),
]

_RULES = [(re.compile(p, re.IGNORECASE), r) for p, r in _RULES_RAW]


# ---------------------------------------------------------------------------
# Deformity detection patterns (for the audit). A trace is "deformed" if any match.
# We deliberately EXCLUDE "task description" from the bare-description detector.
# ---------------------------------------------------------------------------
DEFORMITY_PATTERNS = [
    re.compile(r"\bwe are given a (?:detailed )?description\b", re.IGNORECASE),
    re.compile(r"\bgiven a (?:detailed )?description of (?:an?|the) (?:image|scene|spatial)", re.IGNORECASE),
    re.compile(r"\b(?:from|based on|according to|using|per)\s+(?:this|the|a)\s+(?:scene |detailed )?description\b", re.IGNORECASE),
    re.compile(r"\bin (?:this|the) description\b", re.IGNORECASE),
    re.compile(r"\bmentioned in (?:this|the) (?:scene )?description\b", re.IGNORECASE),
    re.compile(r"(?<!task )(?<!Task )\bthe (?:scene )?description\b", re.IGNORECASE),
    re.compile(r"(?<!task )(?<!Task )\bthis description\b", re.IGNORECASE),
    re.compile(r"\bthe given description\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are) described\b", re.IGNORECASE),
    re.compile(r"\bas described\b", re.IGNORECASE),
    re.compile(r"\bthe text\s+(?:explicitly\s+|clearly\s+|also\s+|only\s+)*"
               r"(?:says|states|mentions|notes|indicates|reads|describes)\b", re.IGNORECASE),
    re.compile(r"\bthe passage\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Whitespace / capitalization normalization
# ---------------------------------------------------------------------------
_CONTRACTIONS = ("the image doesn", "the image don", "the image isn", "the image wasn")


def _normalize(text: str) -> str:
    # "Looking at this image, X..." — the word after the inserted comma was formerly
    # sentence-initial and stayed capitalized; lowercase it so the clause reads well
    # (skip proper-noun-ish all-caps or single-letter tokens, and "I").
    def _lc_after_comma(m):
        w = m.group(1)
        if w == "I" or w.isupper() or len(w) == 1:
            return "Looking at this image, " + w
        return "Looking at this image, " + w[0].lower() + w[1:]
    text = re.sub(r"Looking at this image,\s+([A-Za-z]\w*)", _lc_after_comma, text)
    # "Looking at this image. the X" -> ensure next sentence is capitalized (handled
    # by the generic sentence-cap rule below).
    # Fix sentence capitalization: capitalize first letter after sentence boundary.
    def _cap(m):
        return m.group(1) + m.group(2).upper()
    text = re.sub(r"(^|[.!?]\s+|</think>\s*)([a-z])", _cap, text)
    # collapse internal whitespace runs (but keep newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # remove space before punctuation introduced by deletions
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # fix dangling double colons / "image: :"
    text = re.sub(r":\s*:", ":", text)
    # fix duplicated "the image the image"
    text = re.sub(r"\bthe image the image\b", "the image", text, flags=re.IGNORECASE)
    # tidy spaces around newlines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def _apply_rules(text: str) -> str:
    if not text:
        return text
    for pat, repl in _RULES:
        text = pat.sub(repl, text)
    return _normalize(text)


def clean_think(text: str) -> str:
    """Clean the c2_think reasoning trace."""
    return _apply_rules(text)


def clean_answer(text: str) -> str:
    """Clean the c2_answer prose. Same phrasing fixes; answer decision is untouched
    because no rule edits digits, object labels, or yes/no/direction tokens."""
    return _apply_rules(text)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import os
    import collections

    PHASE_C = ("/mnt/data4/shasta/amar.amarjyoti/research_data/vlm_cot_distill/"
               "phase_c_1060281_deepseekv4_c2_v3_thinking.jsonl")
    TRAIN_IDS = (os.path.join(os.path.dirname(__file__), "..", "..",
                 "subagent_research", "stagec-trace-cleanup", "train_ids.json"))
    TRAIN_IDS = os.path.abspath(TRAIN_IDS)

    train_ids = set(json.load(open(TRAIN_IDS)))

    def count_deformities(text):
        per = {}
        total = 0
        for i, p in enumerate(DEFORMITY_PATTERNS):
            n = len(p.findall(text or ""))
            if n:
                per[i] = n
                total += n
        return total, per

    num_re = re.compile(r"\d+")

    # quoted-phrase core-noun check. We compare the multiset of CONTENT words that
    # appear inside balanced "..." quotes, excluding (a) the small vocabulary our
    # ruleset rewrites and (b) generic stopwords, so the check tracks real object
    # nouns rather than the framing verbs we intentionally change.
    quote_re = re.compile(r'"([^"]{1,120})"')
    _IGNORE = {
        # words our ruleset legitimately introduces/removes
        "description", "image", "scene", "text", "passage", "described", "describes",
        "appears", "appear", "shown", "says", "say", "said", "states", "state",
        "mentions", "mention", "mentioned", "looking", "given", "indicates",
        "provides", "gives", "explicitly", "clearly",
        # stopwords
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "at",
        "and", "or", "as", "it", "this", "that", "we", "be", "by", "with", "from",
        "so", "but", "for", "not", "no", "yes", "than", "also", "thus",
    }

    def core_nouns(text):
        nouns = collections.Counter()
        for q in quote_re.findall(text or ""):
            for w in re.findall(r"[A-Za-z]+", q):
                wl = w.lower()
                if wl not in _IGNORE:
                    nouns[wl] += 1
        return nouns

    # PRIMARY object-preservation guard: the multiset of concrete object/color words
    # over the WHOLE text must be identical before vs after. (The quoted-noun count
    # above is only a secondary heuristic — it drifts harmlessly on traces with
    # unbalanced quotes because quote-pairing re-aligns, so it is reported but not
    # treated as a failure.)
    _OBJ_WORDS = set((
        "car cars truck trucks bus buses van vans motorcycle motorcycles motorbike "
        "vehicle vehicles excavator excavators suv suvs sedan sedans pickup pickups "
        "person adult adults child children worker workers man woman men women people "
        "rider riders pedestrian pedestrians bicycle bicycles bike bikes scooter "
        "scooters train trains pole poles tree trees building buildings sign signs "
        "barrier barriers cone cones trailer trailers crane forklift tractor jeep "
        "taxi cab ambulance dog cat horse "
        "white black red green blue orange yellow silver gray grey brown purple "
        "pink gold beige tan dark light"
    ).split())

    obj_re = re.compile(r"[A-Za-z]+")

    def obj_multiset(text):
        return collections.Counter(
            w.lower() for w in obj_re.findall(text or "") if w.lower() in _OBJ_WORDS
        )

    scanned = 0
    deformed_before = 0
    deformed_after = 0
    before_hits = collections.Counter()
    after_hits = collections.Counter()
    num_violations = 0
    noun_violations = 0
    obj_violations = 0
    examples = []

    for line in open(PHASE_C):
        d = json.loads(line)
        if d["id"] not in train_ids:
            continue
        if not (d.get("parse_ok_c2") and d.get("c2_lands_on_gold")):
            continue
        scanned += 1
        think = d.get("c2_think") or ""
        ans = d.get("c2_answer") or ""

        tb, pb = count_deformities(think)
        ab_ans, pb_ans = count_deformities(ans)
        for i, n in pb.items():
            before_hits[i] += n
        for i, n in pb_ans.items():
            before_hits[i] += n
        if tb + ab_ans > 0:
            deformed_before += 1

        cth = clean_think(think)
        cans = clean_answer(ans)

        ta, pa = count_deformities(cth)
        ta_ans, pa_ans = count_deformities(cans)
        for i, n in pa.items():
            after_hits[i] += n
        for i, n in pa_ans.items():
            after_hits[i] += n
        if ta + ta_ans > 0:
            deformed_after += 1

        # SANITY: number multiset preserved
        if (collections.Counter(num_re.findall(think)) != collections.Counter(num_re.findall(cth))
                or collections.Counter(num_re.findall(ans)) != collections.Counter(num_re.findall(cans))):
            num_violations += 1

        # SANITY (secondary heuristic): quoted core nouns
        if core_nouns(think) != core_nouns(cth) or core_nouns(ans) != core_nouns(cans):
            noun_violations += 1

        # SANITY (primary): object/color word multiset over whole text preserved
        if (obj_multiset(think) != obj_multiset(cth)
                or obj_multiset(ans) != obj_multiset(cans)):
            obj_violations += 1

        if len(examples) < 8 and tb > 0:
            def trunc(s, n=240):
                s = s.replace("\n", " ")
                return s[:n] + ("..." if len(s) > n else "")
            examples.append((d["id"], trunc(think), trunc(cth)))

    print("=" * 78)
    print("STAGE-C DEFORMITY CLEANUP AUDIT")
    print("=" * 78)
    print(f"(a) traces scanned (usable train ids)      : {scanned}")
    print(f"(b) traces w/ >=1 deformity BEFORE         : {deformed_before} "
          f"({100.0*deformed_before/max(scanned,1):.2f}%)")
    print(f"(c) traces w/ >=1 residual deformity AFTER : {deformed_after} "
          f"({100.0*deformed_after/max(scanned,1):.2f}%)")
    print()
    print("(d) per-pattern hit counts  (before -> after):")
    for i, p in enumerate(DEFORMITY_PATTERNS):
        print(f"    [{i:2d}] before={before_hits[i]:>7d}  after={after_hits[i]:>6d}  "
              f"| {p.pattern[:60]}")
    print()
    print("(f) SANITY checks:")
    print(f"    number/coordinate multiset preserved on : "
          f"{scanned - num_violations}/{scanned} "
          f"({'PASS 100%' if num_violations == 0 else f'FAIL ({num_violations} violations)'})")
    print(f"    object/color word multiset preserved on : "
          f"{scanned - obj_violations}/{scanned} "
          f"({'PASS 100%' if obj_violations == 0 else f'FAIL ({obj_violations} violations)'})")
    print(f"    [secondary] quoted core-noun heuristic : "
          f"{scanned - noun_violations}/{scanned} "
          f"({'clean' if noun_violations == 0 else f'{noun_violations} benign quote-realign drifts'})")
    print()
    print("(e) 8 before/after examples (truncated):")
    for eid, b, a in examples:
        print("-" * 78)
        print(f"  id  : {eid}")
        print(f"  BEF : {b}")
        print(f"  AFT : {a}")
    print("=" * 78)
