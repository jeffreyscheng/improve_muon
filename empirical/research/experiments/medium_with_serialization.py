# medium_with_serialization.py - Based on minimal_medium.py with offline logging for gif generation
from empirical.research.training.training_core import *
from empirical.research.training.zeropower import get_zeropower_function
from empirical.research.analysis.online_logging import serialize_model_checkpoint
from empirical.research.analysis.offline_logging import is_logging_step_piecewise_log

args = Hyperparameters()
args.max_minibatches = 1000
loggers = [serialize_model_checkpoint]

model, optimizers, train_loader = create_gpt_with_muon(
    args=args,
    zeropower_fn=get_zeropower_function("newton_schulz", {})
)

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