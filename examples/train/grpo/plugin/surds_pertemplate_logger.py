"""
SURDS per-template binary-accuracy logger for ms-swift GRPO
==========================================================

The native reward path only logs an OVERALL accuracy mean (`rewards/SurdsAccuracy/
mean`). For the per-subtask RL runs (user spec 2026-06-20) we want the binary
correctness of EACH template (yaw / fb / distance / lr / xy2d / depth) tracked
separately, alongside the usual GRPO metrics.

ms-swift can't mask a reward func per template, so this plugin splits the job:

  1. A WEIGHT-0 reward ORM `surds_acc_log` that scores every completion with
     score_surds (binary correct/incorrect), accumulates per-template
     correct/total into a module-level window, and returns all-zeros (so it
     contributes NO gradient — it is a pure side-effect meter).

  2. A TrainerCallback `surds_pertemplate_logger` that, on each `on_log`, reads
     the window on the main process, logs `acc/<template>` + `acc/overall` +
     `n/<template>` to the SAME wandb run at the trainer's global_step, then
     resets the window. Same hook pattern as cognitive_behavior_logger.py.

Wire-up (add to the GRPO job script, alongside the reward plugin):
    --external_plugins "$REPO/examples/train/grpo/plugin/surds_reward_plugin.py" \
                       "$REPO/examples/train/grpo/plugin/surds_pertemplate_logger.py" \
    --reward_funcs <real_reward> surds_acc_log format \
    --reward_weights <w> 0.0 0.2 \
    --callbacks surds_pertemplate_logger

Fully defensive: never raises into the training loop.
"""
import os
import sys
import threading
from collections import defaultdict

from swift.rewards import ORM, orms
from swift.utils import get_logger

try:
    from swift.callbacks import TrainerCallback, callbacks_map
except Exception:  # pragma: no cover
    from transformers import TrainerCallback  # type: ignore
    callbacks_map = None  # type: ignore

logger = get_logger()

# Reuse the reward plugin's score_surds import (same resolution logic).
_SURDS_EVAL_DIR = os.environ.get("SURDS_EVAL_DIR")
if _SURDS_EVAL_DIR is None:
    _THIS = os.path.dirname(os.path.abspath(__file__))
    _SURDS_EVAL_DIR = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..",
                                                    "research", "eval"))
if _SURDS_EVAL_DIR not in sys.path:
    sys.path.insert(0, _SURDS_EVAL_DIR)
try:
    from score_surds import parse_answer, score_one, get_image_wh
    _OK = True
except Exception as _e:  # pragma: no cover
    logger.warning(f"[surds_pertemplate_logger] score_surds import failed ({_e}); "
                   "per-template accuracy will be empty.")
    _OK = False

# ---- module-level windowed accumulators: {template: [n_correct, n_total]} -----
# SEPARATE windows for train rollouts vs held-out eval rollouts. When the job
# carves a validation split (--eval_strategy steps --eval_steps N), the eval
# pass is logged as `val/acc/<tt>` and must NEVER contaminate the live train
# `acc/<tt>` curves — so records are routed by mode at accumulation time.
_LOCK = threading.Lock()
_WINDOWS = {
    "train": defaultdict(lambda: [0, 0]),
    "eval": defaultdict(lambda: [0, 0]),
}

# Set by the callback at construction. Lets the weight-0 ORM tell train from
# eval using the trainer's OWN signal: `model.training`. The GRPO trainer
# determines mode the same way at every reward/metric site
# (`mode = 'train' if self.model.training else 'eval'`, grpo_trainer lines
# 201/240/944/…), and it's False throughout the eval generate+score pass — so
# it's reliable at ORM-call time. (NOTE: `control.should_evaluate` is NOT — it's
# already cleared by the time eval rollouts are scored, which silently routed
# eval records into the train window on the first dense run.)
_TRAINER = None


def _cur_mode():
    try:
        if _TRAINER is not None:
            m = getattr(_TRAINER, "model", None)
            if m is not None and not m.training:
                return "eval"
    except Exception:
        pass
    return "train"


def _record(template_type, correct, mode="train"):
    tt = (template_type or "").strip().lower()
    if not tt:
        return
    win = _WINDOWS.get(mode) or _WINDOWS["train"]
    with _LOCK:
        cell = win[tt]
        cell[0] += 1 if correct else 0
        cell[1] += 1


def _drain(mode="train"):
    """Return {tt: (correct, total)} for `mode`'s window and RESET it."""
    win = _WINDOWS.get(mode) or _WINDOWS["train"]
    with _LOCK:
        snap = {tt: (c, n) for tt, (c, n) in win.items() if n > 0}
        win.clear()
        return snap


class SurdsAccLog(ORM):
    """Weight-0 meter: scores each completion, records per-template correctness,
    returns all-zeros (no gradient contribution)."""

    def __call__(self, completions, **kwargs):
        n = len(completions)
        if not _OK:
            return [0.0] * n
        mode = _cur_mode()   # 'train' | 'eval' — route to the right window
        sols = kwargs.get("solution", [None] * n)
        tts = kwargs.get("template_type", [None] * n)
        imgs = kwargs.get("image_path", [None] * n)
        if not isinstance(sols, (list, tuple)):
            sols = [sols] * n
        if not isinstance(tts, (list, tuple)):
            tts = [tts] * n
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs] * n
        for comp, gold, tt, img in zip(completions, sols, tts, imgs):
            try:
                pred = parse_answer(comp)
                ttl = (tt or "").strip().lower()
                wh = None
                if ttl == "xy2d" and img:
                    try:
                        wh = get_image_wh(img)
                    except Exception:
                        wh = None
                res = score_one(pred, gold, tt, image_wh=wh)
                _record(tt, bool(res.get("correct")), mode)
            except Exception:
                _record(tt, False, mode)
        return [0.0] * n


orms["surds_acc_log"] = SurdsAccLog


class SurdsPerTemplateLogger(TrainerCallback):
    """Drains the per-template window and logs acc/<tt> to wandb at global_step."""

    def __init__(self, args=None, trainer=None):
        self.args = args
        self.trainer = trainer
        global _TRAINER
        _TRAINER = trainer   # so the weight-0 ORM can read control.should_evaluate
        logger.info("[surds_pertemplate_logger] active.")

    def _is_main(self):
        try:
            return bool(self.trainer.accelerator.is_main_process)
        except Exception:
            return True

    def _gather(self, local):
        """Sum per-template (correct, total) across ALL ranks so each logged point
        reflects the whole global batch, not just main's ~1/world_size shard.
        Collective op (all_gather_object) — every rank must reach it, so this is
        called before any early-return. Falls back to `local` for single-process
        runs or on any failure."""
        try:
            import torch.distributed as dist
            if not (dist.is_available() and dist.is_initialized()):
                return local
            world = dist.get_world_size()
            if world <= 1:
                return local
            gathered = [None] * world
            dist.all_gather_object(gathered, local)
            merged = {}
            for part in gathered:
                if not part:
                    continue
                for tt, cn in part.items():
                    cell = merged.setdefault(tt, [0, 0])
                    cell[0] += cn[0]
                    cell[1] += cn[1]
            return {tt: (c, n) for tt, (c, n) in merged.items() if n > 0}
        except Exception as exc:
            logger.warning(f"[surds_pertemplate_logger] cross-rank gather failed: {exc}")
            return local

    def _log(self, mode="train", prefix="acc/"):
        """Drain the `mode` window and log `<base>acc/<tt>` to wandb. base is ''
        for train (→ acc/*) and 'val/' for eval (→ val/acc/*)."""
        try:
            # Drain THIS rank's local window, then all-gather across ranks. ALL ranks
            # must participate in _gather (a collective), so do it before the main-only
            # wandb logging — non-main ranks still contribute their counts and then exit.
            local = _drain(mode)
            snap = self._gather(local)
            if not self._is_main():
                return
            try:
                import wandb
            except Exception:
                return
            if getattr(wandb, "run", None) is None:
                return
            if not snap:
                return
            base = prefix[:-4] if prefix.endswith("acc/") else ""   # '' or 'val/'
            metrics = {}
            tot_c = tot_n = 0
            for tt, (c, n) in snap.items():
                metrics[f"{base}acc/{tt}"] = c / n
                metrics[f"{base}n/{tt}"] = n
                tot_c += c
                tot_n += n
            if tot_n:
                metrics[f"{base}acc/overall"] = tot_c / tot_n
            step = None
            try:
                step = int(self.trainer.state.global_step)
            except Exception:
                pass
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as exc:  # never crash training
            logger.warning(f"[surds_pertemplate_logger] log failed: {exc}")

    def on_log(self, args, state, control, **kwargs):
        # Drain BOTH windows here. on_log is the reliable hook: swift's native
        # eval-metrics logging calls self.log(...) AFTER the eval loop populates
        # the eval window, which fires on_log — so the eval window is full and
        # ready by then. (on_evaluate proved unreliable: either not dispatched to
        # external callbacks, or its wandb.log at the train step got dropped as a
        # step collision. Mode routing itself works — verified eval rollouts do
        # NOT leak into the train `n/` counts.) Two wandb.log calls at the same
        # step merge cleanly; each no-ops when its window is empty.
        self._log("train", "acc/")
        self._log("eval", "val/acc/")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._log("train", "acc/")
        self._log("eval", "val/acc/")
        return control


if callbacks_map is not None:
    callbacks_map["surds_pertemplate_logger"] = SurdsPerTemplateLogger
    logger.info("[surds_pertemplate_logger] registered.")
