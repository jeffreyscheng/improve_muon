from empirical.research.training.training_core import (
    Hyperparameters, setup_distributed_training, setup_logging, log_system_info,
    get_param_groups, warmup_kernels, distributed_data_generator,
    should_validate, validate_and_log, should_terminate, train_step, optimize_step
)
from empirical.research.training.architecture import GPT
from empirical.research.training.spectral_echo import SpectralEcho, update as spectral_echo_update
from pathlib import Path
from functools import partial
from datetime import date
import torch
import torch.distributed as dist


def build_param_name_map(model):
    name_map = {}
    for name, p in model.named_parameters():
        name_map[p] = name
    return name_map


def main():
    args = Hyperparameters()
    args.max_minibatches = 1000

    run_id, rank, world_size, device, master_process = setup_distributed_training()
    print0, run_id_full, logfile = setup_logging(run_id, master_process)

    with open(__file__) as f:
        code = f.read()
    log_system_info(print0, code)

    # Model
    model = GPT(args.vocab_size, 16, 8, 1024, max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if hasattr(m, 'weight') and m.__class__.__name__ == 'Embedding':
            m.bfloat16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    # Param groups
    param_groups = get_param_groups(model)
    # Sanity coverage
    params_collections = [param_groups["hidden"], param_groups["embed"], param_groups["scalar"], param_groups["head"]]
    optimized_parameters_set = {p for params in params_collections for p in params}
    assert optimized_parameters_set == {*model.parameters()}

    # Optimizers
    adam_param_groups = [
        dict(params=param_groups["head"], lr=1/320),
        dict(params=param_groups["embed"], lr=0.3),
        dict(params=param_groups["scalar"], lr=0.015)
    ]
    optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)

    # SpectralEcho for hidden params
    param_to_name = build_param_name_map(model)
    optimizer2 = SpectralEcho(param_groups["hidden"], spectral_echo_update, lr=0.025, momentum=0.95, rank=rank, world_size=world_size, param_to_name=param_to_name)
    optimizers = [optimizer1, optimizer2]

    # Initial LR settings
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # Compile model
    model = torch.compile(model, dynamic=False)

    # Warmup
    warmup_kernels(model, optimizers, args)

    # Data
    train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)

    # Logging (serialization)
    from empirical.research.analysis.logging_utilities import serialize_model_checkpoint
    from empirical.research.analysis.logging_utilities import is_logging_step_piecewise_log
    checkpoint_dir = Path("research_logs/checkpoints")
    run_name = f"spectral_echo_{date.today().strftime('%Y%m%d')}"
    loggers = [partial(serialize_model_checkpoint, run_name=run_name, checkpoint_dir=checkpoint_dir)]

    for step, (inputs, targets) in enumerate(train_loader):
        if should_validate(step, args): validate_and_log(model, step, args, optimizers)
        if should_terminate(step, args): break
        loss = train_step(model, inputs, targets, step, args)
        optimize_step(model, optimizers, step, args)
        if is_logging_step_piecewise_log(step, args.num_iterations):
            from empirical.research.training.training_core import run_loggers
            run_loggers(loggers, model, optimizers, step)

    from empirical.research.training.training_core import _global_print0
    _global_print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

