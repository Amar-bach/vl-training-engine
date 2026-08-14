"""
Cognitive / Reasoning-Behavior Logger for ms-swift GRPO training
================================================================

A time-gated TrainerCallback that periodically classifies recent model
completions for four *cognitive behaviors* that Gandhi et al. 2025
("Cognitive Behaviors that Enable Self-Improving Reasoners", arXiv:2503.01307)
identify as the substrate of self-improving reasoning --

    verification, backtracking, subgoal_setting, branching

-- and logs their prevalence (and reasoning-length stats) to the SAME wandb
run as the training, prefixed `cognitive/`.

The classifier is a lightweight, in-process, regex/keyword matcher. It uses NO
extra model, NO API call, and NO GPU, so it never slows the training loop.

Gating is on WALL-CLOCK time (default 4.5 h, env `COG_LOG_INTERVAL_HOURS`),
NOT on step count, so the cadence is decoupled from throughput. One final log
is forced at training end.


How this hooks into ms-swift (investigated, file:line as of this repo)
----------------------------------------------------------------------
1. external_plugins are *side-effect imported* (no auto-discovery of
   TrainerCallback subclasses):
     swift/arguments/base_args/base_args.py:122  `_import_external_plugins()`
       -> for each path: `import_external_file(external_plugin)`  (line 133-134)
     This runs at argument post-init (base_args.py:159), i.e. BEFORE the
     trainer is built, so a registry mutation at import time is visible later.

2. ms-swift has a first-class custom-callback registry + CLI flag:
     swift/trainers/arguments.py:167   `callbacks: List[str] = field(...)`  (a real --callbacks flag)
     swift/trainers/mixin.py:148-150   `_add_callbacks()`:
         for cb in self.args.callbacks:
             self.add_callback(callbacks_map[cb](self.args, self))
     swift/callbacks/mapping.py:9       `callbacks_map = {...}`  (shared mutable dict)
     swift/callbacks/base.py:9-12       base `TrainerCallback(args, trainer)` -> stores `self.trainer`
   => Registering our class into `callbacks_map` (done at the bottom of this
      file) + passing `--callbacks cognitive_behavior_logger` instantiates it
      with (args, trainer) and adds it via HF's add_callback. The callback can
      then reach the live trainer through `self.trainer`.

3. Recent completions are already buffered in-process on the main process:
     swift/rlhf_trainers/grpo_trainer.py:2125  `self._logs = {...}` with
       `'completion': deque(maxlen=args.generation_batch_size)` (line 2127),
       filled at grpo_trainer.py:282 `self._logs['completion'].extend(...)`.
   We read `trainer._logs['completion']` (decoded strings) -- no extra
   generation, no buffer hook needed. (A ring-buffer fallback via the reward
   plugin is provided but NOT required; see RING-BUFFER FALLBACK below.)

4. wandb is the training run's own client; we push scalars with `wandb.log(...,
   step=global_step)` on the main process only (mirrors grpo_trainer.py:1940).


Wire-up (add to each grpo_bakeoff_*.sh, alongside the reward plugin)
--------------------------------------------------------------------
    --external_plugins "$REPO/examples/train/grpo/plugin/surds_reward_plugin.py" \
                       "$REPO/examples/train/grpo/plugin/cognitive_behavior_logger.py" \
    --callbacks cognitive_behavior_logger \
  (optional)  export COG_LOG_INTERVAL_HOURS=4.5
"""

import os
import re
import time
from typing import List, Optional

try:
    from swift.callbacks import TrainerCallback, callbacks_map
except Exception:  # pragma: no cover - extremely defensive
    from transformers import TrainerCallback  # type: ignore
    callbacks_map = None  # type: ignore

try:
    from swift.utils import get_logger
    logger = get_logger()
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


# ===========================================================================
# Behavior regex lists  --  EDIT THESE FREELY (case-insensitive, matched on
# the reasoning text).  Each list maps a behavior name to its cue patterns.
# Based on Gandhi et al. 2025 taxonomy.
# ===========================================================================
BEHAVIOR_PATTERNS = {
    # Checking / validating an intermediate result.
    "verification": [
        r"let me (?:verify|check|double[- ]?check|confirm)",
        r"let'?s (?:verify|check|double[- ]?check|confirm)",
        r"\bdouble[- ]?check",
        r"\bsanity check\b",
        r"is this correct",
        r"is that correct",
        r"to (?:make|be) sure",
        r"\bmake sure\b",
        r"\bverify(?:ing)?\b",
        r"\bre-?check",
        r"let me re-?compute",
        r"recompute(?:d|s)?\b",
        r"plug(?:ging)? (?:it|this|that|the (?:value|answer)) back",
        r"checking (?:my|the) (?:work|answer|result)",
    ],
    # Abandoning the current line of reasoning and changing course.
    "backtracking": [
        r"\bwait\b",
        r"\bactually\b",
        r"let me reconsider",
        r"on second thought",
        r"\bthat'?s (?:wrong|incorrect|not right)\b",
        r"\bthis is (?:wrong|incorrect|not right)\b",
        r"scratch that",
        r"\bnever ?mind\b",
        r"\bhmm+\b",
        r"let me (?:redo|restart|start over|try again)",
        r"\bgo back\b",
        r"i made (?:a|an) (?:mistake|error)",
        r"\bcorrection\b",
        r"\binstead\b",
    ],
    # Decomposing the problem into ordered steps / subgoals.
    "subgoal_setting": [
        r"\bfirst(?:,|\s+(?:we|i|let'?s|step)\b)",
        r"\bstep\s*\d+\b",
        r"\bstep one\b",
        r"\bnext,? (?:we|i|let'?s)\b",
        r"\bthen,? (?:we|i|let'?s)\b",
        r"break (?:this|it|the problem) (?:down|into)",
        r"let'?s start by",
        r"let me start by",
        r"to (?:solve|do) this,? (?:we|i)",
        r"\bsub-?goal",
        r"the (?:first|next) (?:step|thing) is",
    ],
    # Considering alternative approaches / options.
    "branching": [
        r"\balternatively\b",
        r"another (?:way|approach|option|method|possibility)",
        r"\bwe could also\b",
        r"\bi could also\b",
        r"\boption\s*(?:1|2|3|a|b|c)\b",
        r"on the other hand",
        r"\beither\b.*\bor\b",
        r"one (?:way|approach|option) is",
        r"a (?:different|second) (?:way|approach|option)",
        r"\bor (?:we|i) (?:could|can|might)\b",
    ],
}

# Pre-compile for speed (case-insensitive). One combined pattern per behavior.
_COMPILED = {
    name: re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)
    for name, pats in BEHAVIOR_PATTERNS.items()
}

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)

# Default cadence; overridable per-run via env var.
_DEFAULT_INTERVAL_HOURS = 4.5

# How many recent completions to sample per interval (cap to keep it cheap).
_MAX_SAMPLE = 256

# ---- RING-BUFFER FALLBACK (NOT used by default; here for robustness) -------
# If a future trainer variant does not expose `trainer._logs['completion']`,
# the reward plugin can append decoded completions to this module-level deque
# and we will read from it. To enable, add ONE line at the end of each
# reward ORM `__call__` in surds_reward_plugin.py:
#     from examples.train.grpo.plugin.cognitive_behavior_logger import push_completions
#     push_completions(completions)
# (Left dormant: the in-process `_logs` buffer is the primary path.)
from collections import deque
COMPLETION_RING = deque(maxlen=1024)


def push_completions(completions: List[str]) -> None:
    """Optional hook for the reward plugin (ring-buffer fallback). No-ops safely."""
    try:
        for c in completions:
            if isinstance(c, str):
                COMPLETION_RING.append(c)
    except Exception:
        pass


# ===========================================================================
# Classification
# ===========================================================================
def _reasoning_span(text: str) -> str:
    """Return the <think>...</think> span if present, else the full text."""
    if not isinstance(text, str):
        return ""
    m = _THINK_RE.search(text)
    if m:
        return m.group(1)
    return text


def classify_completion(text: str) -> dict:
    """
    Classify a single completion's reasoning text.
    Returns {behavior: bool, ...} plus 'total' (int count of behaviors present).
    """
    reasoning = _reasoning_span(text)
    flags = {}
    total = 0
    for name, rx in _COMPILED.items():
        present = bool(rx.search(reasoning))
        flags[name] = present
        total += int(present)
    flags["total"] = total
    flags["_reasoning"] = reasoning  # transient, used for length stats
    return flags


# ===========================================================================
# The callback
# ===========================================================================
class CognitiveBehaviorLogger(TrainerCallback):
    """
    Time-gated reasoning-behavior logger. Instantiated by ms-swift as
    `callbacks_map['cognitive_behavior_logger'](args, trainer)`.
    """

    def __init__(self, args=None, trainer=None):
        # Match swift.callbacks.base.TrainerCallback(args, trainer) signature,
        # but stay tolerant if instantiated bare.
        self.args = args
        self.trainer = trainer
        try:
            self.interval_hours = float(
                os.environ.get("COG_LOG_INTERVAL_HOURS", _DEFAULT_INTERVAL_HOURS)
            )
        except Exception:
            self.interval_hours = _DEFAULT_INTERVAL_HOURS
        self._last_log_t = time.time()
        self._final_done = False
        logger.info(
            f"[cognitive_behavior_logger] active; interval="
            f"{self.interval_hours:.2f}h (env COG_LOG_INTERVAL_HOURS)."
        )

    # -- helpers -----------------------------------------------------------
    def _is_main_process(self) -> bool:
        try:
            return bool(self.trainer.accelerator.is_main_process)
        except Exception:
            # If we can't tell, fall back to torch/global rank 0.
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    return dist.get_rank() == 0
            except Exception:
                pass
            return True

    def _global_step(self) -> Optional[int]:
        try:
            return int(self.trainer.state.global_step)
        except Exception:
            return None

    def _recent_completions(self) -> List[str]:
        """Primary: trainer._logs['completion']. Fallback: module ring buffer."""
        comps: List[str] = []
        try:
            logs = getattr(self.trainer, "_logs", None)
            if logs is not None and logs.get("completion"):
                comps = [c for c in logs["completion"] if isinstance(c, str)]
        except Exception:
            comps = []
        if not comps and COMPLETION_RING:
            comps = [c for c in COMPLETION_RING if isinstance(c, str)]
        if len(comps) > _MAX_SAMPLE:
            comps = comps[-_MAX_SAMPLE:]
        return comps

    def _maybe_log(self, force: bool = False) -> None:
        """Run classification + wandb log if interval elapsed (or forced)."""
        try:
            now = time.time()
            elapsed_h = (now - self._last_log_t) / 3600.0
            if not force and elapsed_h < self.interval_hours:
                return
            if not self._is_main_process():
                # Still advance the clock so we don't spin every step.
                self._last_log_t = now
                return

            # wandb must be live; no-op gracefully otherwise.
            try:
                import wandb
            except Exception:
                self._last_log_t = now
                return
            if getattr(wandb, "run", None) is None:
                self._last_log_t = now
                return

            comps = self._recent_completions()
            n = len(comps)
            if n == 0:
                # Nothing to classify yet; reset clock and move on.
                self._last_log_t = now
                return

            counts = {b: 0 for b in BEHAVIOR_PATTERNS}
            total_behaviors = 0
            total_chars = 0
            total_tokens = 0  # whitespace-split proxy
            for c in comps:
                res = classify_completion(c)
                for b in BEHAVIOR_PATTERNS:
                    counts[b] += int(res[b])
                total_behaviors += res["total"]
                reasoning = res["_reasoning"]
                total_chars += len(reasoning)
                total_tokens += len(reasoning.split())

            metrics = {f"cognitive/{b}_frac": counts[b] / n for b in BEHAVIOR_PATTERNS}
            metrics.update({
                f"cognitive/{b}_count": counts[b] for b in BEHAVIOR_PATTERNS
            })
            metrics["cognitive/mean_behaviors_per_completion"] = total_behaviors / n
            metrics["cognitive/mean_reasoning_chars"] = total_chars / n
            metrics["cognitive/mean_reasoning_tokens"] = total_tokens / n
            metrics["cognitive/n_completions"] = n
            metrics["cognitive/elapsed_hours_since_last_log"] = elapsed_h

            step = self._global_step()
            try:
                if step is not None:
                    wandb.log(metrics, step=step)
                else:
                    wandb.log(metrics)
            except Exception as exc:
                logger.warning(f"[cognitive_behavior_logger] wandb.log failed: {exc}")

            logger.info(
                f"[cognitive_behavior_logger] logged at step={step} "
                f"(n={n}, elapsed={elapsed_h:.2f}h): "
                + ", ".join(f"{b}={counts[b] / n:.2f}" for b in BEHAVIOR_PATTERNS)
            )
            self._last_log_t = now
        except Exception as exc:  # never crash training
            logger.warning(f"[cognitive_behavior_logger] _maybe_log error: {exc}")

    # -- HF TrainerCallback hooks -----------------------------------------
    def on_log(self, args, state, control, **kwargs):
        # Cheap: only fires on logging_steps cadence; we still gate on wall-clock.
        self._maybe_log(force=False)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        # Backup trigger in case on_log is sparse; wall-clock gate keeps it cheap.
        self._maybe_log(force=False)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self._final_done:
            self._final_done = True
            self._maybe_log(force=True)
        return control


# ===========================================================================
# Registration (side-effect at import; external_plugins imports this file).
#   --callbacks cognitive_behavior_logger  -> callbacks_map[...] (args, trainer)
# ===========================================================================
if callbacks_map is not None:
    callbacks_map["cognitive_behavior_logger"] = CognitiveBehaviorLogger
    logger.info(
        "[cognitive_behavior_logger] registered in callbacks_map as "
        "'cognitive_behavior_logger'."
    )
