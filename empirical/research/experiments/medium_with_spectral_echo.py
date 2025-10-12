from empirical.research.training.training_core import (
    Hyperparameters, create_gpt_with_optimizer, create_train_loader,
    should_validate, validate_and_log, should_terminate, train_step, optimize_step, run_loggers, _global_print0
)
from empirical.research.training.spectral_echo import SpectralEcho, update as spectral_echo_update
from empirical.research.analysis.logging_utilities import serialize_model_checkpoint, is_logging_step_piecewise_log
from pathlib import Path
from functools import partial
from datetime import date
import torch


def build_hidden_optimizer_spectral_echo(params, *, model, param_to_name, device, rank, world_size, lr, weight_decay, momentum):
    return SpectralEcho(params, spectral_echo_update, lr=lr, momentum=momentum, rank=rank, world_size=world_size, param_to_name=param_to_name)


args = Hyperparameters(); args.max_minibatches = 1000
model, optimizers = create_gpt_with_optimizer(args=args, build_hidden_optimizer_fn=build_hidden_optimizer_spectral_echo)
train_loader = create_train_loader(args)

# Logging (serialization)
checkpoint_dir = Path("research_logs/checkpoints")
run_name = f"spectral_echo_{date.today().strftime('%Y%m%d')}"
loggers = [partial(serialize_model_checkpoint, run_name=run_name, checkpoint_dir=checkpoint_dir)]

for step, (inputs, targets) in enumerate(train_loader):
    if should_validate(step, args): validate_and_log(model, step, args, optimizers)
    if should_terminate(step, args): break
    loss = train_step(model, inputs, targets, step, args)
    optimize_step(model, optimizers, step, args)
    if is_logging_step_piecewise_log(step, args.num_iterations):
        run_loggers(loggers, model, optimizers, step)

_global_print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
