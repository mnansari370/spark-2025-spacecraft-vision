"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time
import json
import datetime
import os

import torch

from ..misc import dist_utils, profiler_utils
from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate

try:
    import wandb
except Exception:
    wandb = None


def _to_float(x):
    """Safely convert tensors / numpy scalars / python numbers to float."""
    if x is None:
        return None
    try:
        # torch tensor -> python float
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return None


class DetSolver(BaseSolver):

    def fit(self):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"number of trainable parameters: {n_parameters}")

        # ---------------- W&B init (main process only) ----------------
        self._use_wandb = False
        self._wandb_run = None

        if wandb is not None and dist_utils.is_main_process():
            # Only enable if user configured it via env vars
            if os.environ.get("WANDB_PROJECT") or os.environ.get("WANDB_MODE"):
                try:
                    self._wandb_run = wandb.init(
                        project=os.environ.get("WANDB_PROJECT", "CV_Detection"),
                        entity=os.environ.get("WANDB_ENTITY", None),
                        name=os.environ.get("WANDB_NAME", None),
                        config={
                            "config_path": str(getattr(args, "config", "")),
                            "epoches": int(getattr(args, "epoches", -1)),
                            "output_dir": str(getattr(args, "output_dir", "")),
                        },
                    )
                    print("WANDB_URL:", self._wandb_run.url)
                    self._use_wandb = True
                except Exception as e:
                    print("[WARN] wandb.init failed:", repr(e))
                    self._use_wandb = False
                    self._wandb_run = None

        best_stat = {"epoch": -1}

        start_time = time.time()
        start_epoch = self.last_epoch + 1

        try:
            for epoch in range(start_epoch, args.epoches):

                self.train_dataloader.set_epoch(epoch)
                if dist_utils.is_dist_available_and_initialized():
                    self.train_dataloader.sampler.set_epoch(epoch)

                train_stats = train_one_epoch(
                    self.model,
                    self.criterion,
                    self.train_dataloader,
                    self.optimizer,
                    self.device,
                    epoch,
                    max_norm=args.clip_max_norm,
                    print_freq=args.print_freq,
                    ema=self.ema,
                    scaler=self.scaler,
                    lr_warmup_scheduler=self.lr_warmup_scheduler,
                    writer=self.writer,
                )

                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

                self.last_epoch += 1

                # save last/checkpoint
                if self.output_dir:
                    checkpoint_paths = [self.output_dir / "last.pth"]
                    if (epoch + 1) % args.checkpoint_freq == 0:
                        checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                    for checkpoint_path in checkpoint_paths:
                        dist_utils.save_on_master(self.state_dict(), checkpoint_path)

                module = self.ema.module if self.ema else self.model
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    self.device,
                )

                # Tensorboard writer + best.pth logic
                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        vals = test_stats[k]
                        if isinstance(vals, (list, tuple)):
                            for i, v in enumerate(vals):
                                fv = _to_float(v)
                                if fv is not None:
                                    self.writer.add_scalar(f"Test/{k}_{i}", fv, epoch)

                    # best selection uses index 0
                    v0 = test_stats[k][0] if isinstance(test_stats[k], (list, tuple)) else test_stats[k]
                    v0f = _to_float(v0)

                    if k in best_stat:
                        if v0f is not None and v0f > best_stat[k]:
                            best_stat["epoch"] = epoch
                        if v0f is not None:
                            best_stat[k] = max(best_stat[k], v0f)
                    else:
                        best_stat["epoch"] = epoch
                        best_stat[k] = v0f if v0f is not None else -1e18

                    if best_stat["epoch"] == epoch and self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best.pth")

                print(f"best_stat: {best_stat}")

                # ---------------- W&B logging (main process only) ----------------
                if self._use_wandb and dist_utils.is_main_process():
                    wb = {}

                    # train stats
                    for k, v in train_stats.items():
                        fv = _to_float(v)
                        if fv is not None:
                            wb[f"train/{k}"] = fv

                    # test stats (log first metric as main)
                    for k, v in test_stats.items():
                        if isinstance(v, (list, tuple)) and len(v) > 0:
                            fv = _to_float(v[0])
                        else:
                            fv = _to_float(v)
                        if fv is not None:
                            wb[f"val/{k}"] = fv

                    # LR
                    try:
                        wb["lr"] = float(self.optimizer.param_groups[0]["lr"])
                    except Exception:
                        pass

                    # best
                    if "coco_eval_bbox" in best_stat and best_stat["coco_eval_bbox"] is not None:
                        wb["best/coco_eval_bbox"] = float(best_stat["coco_eval_bbox"])
                        wb["best/epoch"] = int(best_stat["epoch"])

                    wb["epoch"] = int(epoch)

                    try:
                        wandb.log(wb, step=int(epoch))
                    except Exception as e:
                        print("[WARN] wandb.log failed:", repr(e))

                # local log.txt
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

                if self.output_dir and dist_utils.is_main_process():
                    with (self.output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    # save evaluator dump
                    if coco_evaluator is not None:
                        (self.output_dir / "eval").mkdir(exist_ok=True)
                        if "bbox" in coco_evaluator.coco_eval:
                            filenames = ["latest.pth"]
                            if epoch % 50 == 0:
                                filenames.append(f"{epoch:03}.pth")
                            for name in filenames:
                                torch.save(
                                    coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name,
                                )

        finally:
            # Always finish W&B cleanly
            if self._use_wandb and dist_utils.is_main_process():
                try:
                    wandb.finish()
                except Exception:
                    pass

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")

    def val(self):
        self.eval()
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
        )

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        return
