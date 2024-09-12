import torch.distributed as dist
from rich.console import Console
from rich.table import Table

def calculate_optimizer_config_in_distributed_setting(trainer, warmup_steps, learning_rate, weight_decay):
    num_devices = max(1, trainer.num_devices)
    dataset_size = len(trainer.datamodule.datasets['train'])
    
    train_batch_size = trainer.datamodule.gen_batch_size
    effective_batch_size = train_batch_size * trainer.accumulate_grad_batches * num_devices

    total_training_steps = (dataset_size // effective_batch_size) * trainer.max_epochs
    total_warmup_steps = warmup_steps // (trainer.accumulate_grad_batches * num_devices)
    training_steps_per_epoch = dataset_size // effective_batch_size
    
    scaled_learning_rate = learning_rate * num_devices
    
    if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
        _print_config_table(
            total_training_steps, training_steps_per_epoch, weight_decay,
            trainer.max_epochs, total_warmup_steps, learning_rate, scaled_learning_rate
        )
    
    return total_training_steps, total_warmup_steps, scaled_learning_rate

def _print_config_table(total_steps, steps_per_epoch, weight_decay, epochs, warmup_steps, base_lr, scaled_lr):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    rows = [
        ("Total training steps", total_steps),
        ("Training steps per epoch", steps_per_epoch),
        ("Weight decay", weight_decay),
        ("Number of epochs", epochs),
        ("Total warmup steps", warmup_steps),
        ("Base learning rate", base_lr),
        ("Scaled learning rate", scaled_lr)
    ]

    for row in rows:
        table.add_row(row[0], str(row[1]))

    console.print(table)