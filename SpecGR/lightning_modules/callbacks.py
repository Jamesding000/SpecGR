import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, Callback
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

def custom_metrics_format(value):
    if value == int(value):
        return f"{int(value)}"
    return f"{value:.3f}"

NICE_PROGRESS_BAR = RichProgressBar(
    theme=RichProgressBarTheme(
        description="dark_orange3",  # Darker color for description
        progress_bar="medium_spring_green",  # A lighter shade of green
        progress_bar_finished="green1",  # Existing green
        progress_bar_pulse="dark_magenta",  # Darker purple for pulse
        batch_progress="spring_green3",  # Existing purple
        time="#0080FF",  # Blue
        processing_speed="#FFD700",  # Yellow
        metrics="#8000FF",  # Bright purple
        metrics_text_delimiter="\n",
        metrics_format=".4f",  # Custom format function
    ),
    # refresh_rate = 100
)
