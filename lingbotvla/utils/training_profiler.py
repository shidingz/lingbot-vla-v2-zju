"""Opt-in training phase ranges and native Ascend/CUDA traces."""
from contextlib import nullcontext
from functools import wraps
from pathlib import Path

import torch


class TrainingProfiler:
    def __init__(self, args):
        ranks = args.profile_ranks.strip()
        self.selected = args.enable_profiling and (
            ranks == "all" or args.global_rank in {int(r) for r in ranks.split(",") if r.strip()}
        )
        self.start_step = args.profile_start_step
        self.end_step = args.profile_end_step
        self.current_step = 0
        self.profiler = None
        if not self.selected:
            return
        if not 1 <= self.start_step < self.end_step:
            raise ValueError("Profiling requires 1 <= profile_start_step < profile_end_step")
        trace_dir = Path(args.profile_trace_dir) / f"rank{args.global_rank}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        if torch._C._get_privateuse1_backend_name() == "npu":
            import torch_npu
            api = torch_npu.profiler
            activities = [api.ProfilerActivity.CPU, api.ProfilerActivity.NPU]
            extra = {"experimental_config": api._ExperimentalConfig(
                profiler_level=api.ProfilerLevel.Level1,
                aic_metrics=api.AiCMetrics.PipeUtilization,
                data_simplification=False,
            )}
        else:
            api = torch.profiler
            activities = [api.ProfilerActivity.CPU, api.ProfilerActivity.CUDA]
            extra = {}
        warmup = int(self.start_step > 1)
        self.profiler = api.profile(
            activities=activities,
            schedule=api.schedule(wait=self.start_step - warmup - 1, warmup=warmup,
                                  active=self.end_step - self.start_step, repeat=1),
            on_trace_ready=api.tensorboard_trace_handler(str(trace_dir)),
            record_shapes=args.profile_record_shapes,
            profile_memory=args.profile_profile_memory,
            with_stack=args.profile_with_stack,
            **extra,
        )

    def scope(self, name):
        if self.selected and self.start_step - 1 <= self.current_step < self.end_step:
            return torch.profiler.record_function(name)
        return nullcontext()

    def instrument_model(self, model):
        if not self.selected:
            return
        # Wrapping forward also captures activation checkpoint recomputation.
        for module in model.modules():
            if module.__class__.__name__ == "Qwen2FusedExperts":
                original = module.forward

                @wraps(original)
                def forward(*args, _original=original, **kwargs):
                    with self.scope("model.moe_experts"):
                        return _original(*args, **kwargs)

                module.forward = forward

    def start(self):
        if self.profiler is not None:
            self.profiler.start()

    def step(self):
        if self.profiler is not None:
            self.profiler.step()

    def stop(self):
        if self.profiler is not None:
            self.profiler.stop()
