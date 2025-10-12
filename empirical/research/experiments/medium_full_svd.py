# medium_full_svd.py - Medium experiment with full SVD polar backend and serialization
from empirical.research.training.training_core import (
    Hyperparameters, create_gpt_with_optimizer, create_train_loader,
    should_validate, validate_and_log, train_step, optimize_step, _global_print0
)
from empirical.research.training.zeropower import get_zeropower_function, make_update_function
from empirical.research.training.muon import Muon
from empirical.research.analysis.logging_utilities import serialize_model_checkpoint
from empirical.research.analysis.logging_utilities import is_logging_step_piecewise_log
from pathlib import Path
from functools import partial
from datetime import date
import torch

args = Hyperparameters(); args.max_minibatches = 1000

def build_hidden_optimizer_muon(params, *, model, param_to_name, device, rank, world_size, lr, weight_decay, momentum):
    update_fn = make_update_function(get_zeropower_function("svd_polar", {}))
    return Muon(params, update_fn, lr=lr, momentum=momentum, rank=rank, world_size=world_size)

model, optimizers = create_gpt_with_optimizer(args=args, build_hidden_optimizer_fn=build_hidden_optimizer_muon)
train_loader = create_train_loader(args)

checkpoint_dir = Path("logs")
run_name = f"medium_full_svd_{date.today().strftime('%Y%m%d')}"
loggers = [partial(serialize_model_checkpoint, run_name=run_name, checkpoint_dir=checkpoint_dir)]

for step, (inputs, targets) in enumerate(train_loader):
    if should_validate(step, args): validate_and_log(model, step, args, optimizers)
    if step >= args.num_iterations: break
    loss = train_step(model, inputs, targets, step, args)
    optimize_step(model, optimizers, step, args)
    if is_logging_step_piecewise_log(step, args.num_iterations): run_loggers(loggers, model, optimizers, step)

# Final cleanup - access the global print function
_global_print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

import torch.distributed as dist
dist.destroy_process_group()
