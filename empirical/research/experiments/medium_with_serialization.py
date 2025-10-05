# medium_with_serialization.py - Based on minimal_medium.py with offline logging for gif generation
from empirical.research.training.training_core import *
from empirical.research.training.zeropower import get_zeropower_function
from empirical.research.analysis.logging_utilities import serialize_model_checkpoint
from empirical.research.analysis.logging_utilities import is_logging_step_piecewise_log
from pathlib import Path
from functools import partial
from datetime import date

args = Hyperparameters()
args.max_minibatches = 1000

model, optimizers, train_loader = create_gpt_with_muon(
    args=args,
    zeropower_fn=get_zeropower_function("newton_schulz", {})
)

# Configure explicit checkpoint directory and run name
checkpoint_dir = Path("research_logs/checkpoints")
run_name = f"medium_{date.today().strftime('%Y%m%d')}"
loggers = [partial(serialize_model_checkpoint, run_name=run_name, checkpoint_dir=checkpoint_dir)]

for step, (inputs, targets) in enumerate(train_loader):
    if should_validate(step, args): validate_and_log(model, step, args, optimizers)
    if should_terminate(step, args): break
    
    loss = train_step(model, inputs, targets, step, args)
    optimize_step(model, optimizers, step, args)
    if is_logging_step_piecewise_log(step, args.num_iterations): run_loggers(loggers, model, optimizers, step)

# Final cleanup - access the global print function
from empirical.research.training.training_core import _global_print0
_global_print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

import torch.distributed as dist
dist.destroy_process_group()
