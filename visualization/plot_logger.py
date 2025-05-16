"""
PlotLogger class for creating a dashboard of plots during model training.
"""
import os
import io
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from pytorch_lightning import Callback
from typing import List, Dict, Optional, Any
from IPython.display import Image, display
from PIL import Image as PILImage
from concurrent.futures import ThreadPoolExecutor

from .plot_utils import BasePlot, PlotUtils


class PlotLogger(Callback):
    """
    A PyTorch Lightning callback that creates a dashboard of plots during training.
    
    This logger collects outputs from the validation step, generates plots,
    saves them to disk, logs them to TensorBoard, and creates an animation
    of the plots over time.
    """
    
    def __init__(
        self,
        plots: List[BasePlot],
        save_path: str = "",
        interval_ms: int = 500,
        export_filename: str = "training_animation",
        log_to_tensorboard: bool = True,
        max_gif_frames: Optional[int] = None,
        show_final_plots: bool = False,
        show_gif: bool = True,
        verbose: bool = False
    ):
        super().__init__()
        self.plots = {plot.name: plot for plot in plots}
        self.save_path = save_path
        self.interval_ms = interval_ms
        self.export_filename = export_filename
        self.log_to_tensorboard = log_to_tensorboard
        self.max_gif_frames = max_gif_frames
        self.show_gif = show_gif
        self.verbose = verbose
        self.show_final_plots = show_final_plots
        self._val_outputs = []
        self._epoch_images = []
        self._executor = None
        self.current_epoch = 0

        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._val_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._val_outputs:
            return

        epoch = trainer.current_epoch
        self.current_epoch = epoch

        outputs = self._gather_outputs()
        plot_data = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in outputs.items()}

        images_dict = {}
        for name, plot in self.plots.items():
            try:
                fig = plot.plot(plot_data)

                if plot.config.backend == "matplotlib":
                    fig.patch.set_facecolor('white')

                if self.log_to_tensorboard and trainer.logger and hasattr(trainer.logger.experiment, "add_image"):
                    img_tensor = PlotUtils.fig_to_tensor(fig, plot.config.backend)
                    trainer.logger.experiment.add_image(f"plots/{name}", img_tensor, global_step=epoch)

                if plot.config.backend == "matplotlib":
                    fig.canvas.draw()
                    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                else:
                    img_bytes = fig.to_image(format="png", scale=2)
                    img = np.array(PILImage.open(io.BytesIO(img_bytes)).convert("RGB"))

                images_dict[name] = img

                if self.save_path:
                    if self._executor is None:
                        self._executor = ThreadPoolExecutor(max_workers=4)
                    filepath = f"{self.save_path}/{name}_epoch_{epoch:04d}.png"
                    self._executor.submit(PlotUtils.save_figure, fig, filepath, plot.config.backend)

                if plot.config.backend == "matplotlib":
                    plt.close(fig)

            except Exception as e:
                if self.verbose:
                    print(f"[PlotLogger] Failed to process {name}: {e}")

        self._epoch_images.append((epoch, images_dict))
        self._val_outputs = []

    def on_train_end(self, trainer, pl_module):
        if self._executor:
            self._executor.shutdown(wait=True)

        if not self._epoch_images:
            return
        if self.show_final_plots:
            self._show_final_plots()

        if self.show_gif and len(self._epoch_images) > 1:
            gif_path = self._export_animation()
            try:
                display(Image(filename=gif_path))
            except Exception:
                if self.verbose:
                    print(f"[PlotLogger] Animation saved to {gif_path}")

    def _gather_outputs(self) -> Dict[str, Any]:
        if not self._val_outputs:
            return {}

        all_keys = {k for plot in self.plots.values() for k in plot.required_keys}
        available_keys = set(self._val_outputs[0].keys())
        keys_to_collect = all_keys.intersection(available_keys)

        result = {}
        for k in keys_to_collect:
            try:
                values = [out[k] for out in self._val_outputs if k in out]

                if values and all(torch.is_tensor(v) for v in values):
                    shapes = [v.shape[1:] for v in values]
                    if all(s == shapes[0] for s in shapes):
                        result[k] = torch.cat(values, dim=0)
                    else:
                        if self.verbose:
                            print(f"[PlotLogger] Inconsistent shapes for key '{k}': {shapes}")
                        result[k] = values[-1]
                else:
                    result[k] = values[-1]
            except Exception as e:
                if self.verbose:
                    print(f"[PlotLogger] Error collecting '{k}': {e}")

        return result

    def _show_final_plots(self):
        _, final_images = self._epoch_images[-1]
        for name, plot in self.plots.items():
            print(f"{name} (Epoch {self.current_epoch})")

            if plot.config.backend == "plotly":
                fig = plot.plot({k: v for k, v in final_images.items()})  # you might need to pass raw data
                fig.show()  # or display(fig)
            else:  # matplotlib
                fig = plot.plot({k: v for k, v in final_images.items()})
                fig.show()


    def _export_animation(self) -> str:
        gif_path = os.path.join(self.save_path or ".", f"{self.export_filename}.gif")
        frames = self._epoch_images
        if self.max_gif_frames and len(frames) > self.max_gif_frames:
            idx = np.linspace(0, len(frames) - 1, self.max_gif_frames, dtype=int)
            frames = [frames[i] for i in idx]

        num_plots = len(self.plots)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]

        img_objs = [
            ax.imshow(frames[0][1][name])
            for ax, name in zip(axes, self.plots.keys())
        ]

        for ax, name in zip(axes, self.plots.keys()):
            ax.set_title(f"{name} (Epoch {frames[0][0]})")
            ax.axis("off")

        def update(i):
            epoch, imgs = frames[i]
            for img_obj, ax, (name, _) in zip(img_objs, axes, self.plots.items()):
                img_obj.set_data(imgs[name])
                ax.set_title(f"{name} (Epoch {epoch})")

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=self.interval_ms)
        ani.save(gif_path, writer=PillowWriter(fps=1000 // self.interval_ms))
        plt.close(fig)

        if self.verbose:
            print(f"[PlotLogger] Exported GIF to {gif_path}")
        return gif_path
