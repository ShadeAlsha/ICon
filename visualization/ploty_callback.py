from IPython.display import display, clear_output, HTML
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pytorch_lightning as pl
import plotly.io as pio
from PIL import Image
import io
from typing import Dict, Optional
from matplotlib.colors import ListedColormap
from dataclasses import dataclass
from typing import Tuple, Union, List
from collections import defaultdict
import ipywidgets as widgets


# Custom colormap
high_contrast_colors = [
    "#7F3C8D", "#1CA878", "#3969AC", "#F2B701", "#D63F6C",
    "#A0C95A", "#E68310", "#008695", "#CF1C90", "#005082"
]
custom_cmap = ListedColormap(high_contrast_colors)

@dataclass
class PlotConfig:
    show_plots: bool = False
    selected_plots: List[str] = None
    figure_size: Tuple[int, int] = (5, 5)
    dpi: int = 150
    cmap = custom_cmap
    def __post_init__(self):
        if self.selected_plots is None:
            self.selected_plots = ['embeddings', 'cluster_sizes']

class BasePlot:
    def __init__(self, config: PlotConfig):
        self.config = config

    def plot(self, data: Dict) -> go.Figure:
        raise NotImplementedError

class EmbeddingsPlot(BasePlot):
    def plot(self, data: Dict) -> go.Figure:
        embeddings = np.array(data['embeddings'])
        labels = np.array(data['labels'])

        if embeddings.shape[1] == 2:
            return self.plot_2d(embeddings, labels)
        elif embeddings.shape[1] == 3:
            return self.plot_3d(embeddings, labels)
        else:
            raise ValueError("Embedding dimension must be 2 or 3.")

    def plot_2d(self, embeddings: np.ndarray, labels: np.ndarray) -> go.Figure:
        labels_str = labels.astype(str)
        # Sort data by label to ensure consistent color assignment
        sorted_indices = np.argsort(labels_str)

        fig = px.scatter(
            x=embeddings[sorted_indices, 0],
            y=embeddings[sorted_indices, 1],
            color=labels_str[sorted_indices],
            color_discrete_sequence=high_contrast_colors,            
        )

        fig.update_layout(
            title="2D Embeddings Visualization",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            template="simple_white",
            legend_title="Label"
        )
        fig.update_traces(marker=dict(size=4, opacity=0.5))
        return fig

    def plot_3d(self, embeddings: np.ndarray, labels: np.ndarray) -> go.Figure:
        fig = go.Figure()
        labels_str = labels.astype(str)
        unique_labels = sorted(np.unique(labels_str), key=lambda x: int(x) if x.isdigit() else x)

        for i, label in enumerate(unique_labels):
            mask = labels_str == label
            fig.add_trace(go.Scatter3d(
                x=embeddings[mask, 0],
                y=embeddings[mask, 1],
                z=embeddings[mask, 2],
                mode='markers',
                name=label,
                marker=dict(
                    size=4,
                    opacity=0.8,
                    color=high_contrast_colors[i % len(high_contrast_colors)]
                )
            ))

        # Optional: grid lines
        u = np.linspace(0, 2 * np.pi, 36)
        v = np.linspace(0, np.pi, 18)
        for phi in v:
            x = np.cos(u) * np.sin(phi)
            y = np.sin(u) * np.sin(phi)
            z = np.full_like(u, np.cos(phi))
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='gray', width=1), showlegend=False))
        for theta in u:
            x = np.cos(theta) * np.sin(v)
            y = np.sin(theta) * np.sin(v)
            z = np.cos(v)
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='gray', width=1), showlegend=False))

        fig.update_layout(
            title="3D Embeddings on Unit Sphere",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
            ),
            template="simple_white",
            showlegend=True,
            legend_title="Label"
        )
        return fig


class ClusterSizesPlot(BasePlot):
    def plot(self, data: Dict) -> go.Figure:
        probabilities = torch.tensor(data['probabilities'])
        cluster_sizes = probabilities.sum(dim=0)
        uncertainty = torch.sqrt((probabilities * (1 - probabilities)).sum(dim=0) / probabilities.size(0))

        cluster_sizes, indices = torch.sort(cluster_sizes, descending=True)
        uncertainty = uncertainty[indices]

        fig = go.Figure(data=[
            go.Bar(
                x=np.arange(len(cluster_sizes)),
                y=cluster_sizes.numpy(),
                error_y=dict(type='data', array=uncertainty.numpy(), visible=True),
                marker_color='skyblue'
            )
        ])

        fig.update_layout(
            title='Cluster Sizes with Uncertainty',
            xaxis_title='Cluster Index',
            yaxis_title='Number of Points',
            template='simple_white'
        )
        return fig

class NeighborhoodDistPlot(BasePlot):
    def plot(self, data: Dict) -> go.Figure:
        learned = torch.nn.functional.normalize(data['learned_distribution'], p=1, dim=-1)
        supervisory = torch.nn.functional.normalize(data['supervisory_distribution'], p=1, dim=-1)

        learned = torch.clamp(learned, min=1e-8)
        supervisory = torch.clamp(supervisory, min=1e-8)
        _, indices = torch.sort(supervisory + 1e-4 * learned, dim=-1)

        probs_sorted = self._gather_and_process(learned, indices)
        target_sorted = self._gather_and_process(supervisory, indices)

        x = np.arange(probs_sorted[0].size(0))[::-1]
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x,
            y=target_sorted[0].numpy(),
            mode='markers',
            name='Target Distribution P',
            marker=dict(color='orange', size = 2)
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=probs_sorted[0].numpy(),
            mode='markers',
            name='Learned Distribution Q',
            marker=dict(color='blue', size = 2)
        ))
        fig.update_layout(
            title='Neighbor Selection Probability Distributions',
            xaxis_title='Neighbors Ordered by Proximity',
            yaxis_title='Selection Probability',
            yaxis_type='log',
            template='simple_white',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.0,
                xanchor='center',
                x=0.5,
                font=dict(size=10),
                itemwidth=30
            ),
            margin=dict(t=80, b=40, l=40, r=20)
        )

        return fig

    def _gather_and_process(self, kernel: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gathered = torch.gather(kernel, 1, indices)
        mean = gathered.mean(dim=0)
        q1, q3 = torch.quantile(gathered, torch.tensor([0.25, 0.75]), dim=0)
        iqr = q3 - q1
        return mean.flip(0), iqr.flip(0)


class ProbabilitiesStarPlot(BasePlot):
    def plot(self, data: Dict) -> go.Figure:
        probs = torch.tensor(data['probabilities'])
        labels = torch.tensor(data['labels'])
        n_clusters = probs.shape[1]

        theta = torch.linspace(0, 2 * torch.pi, n_clusters + 1)[:-1]
        vertices = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        points = probs @ vertices

        fig = go.Figure()
        for i in range(n_clusters):
            mask = (labels == i)
            fig.add_trace(go.Scatter(
                x=points[mask, 0],
                y=points[mask, 1],
                mode='markers',
                name=f'Cluster {i}',
                marker=dict(size=10, color=high_contrast_colors[i % len(high_contrast_colors)], opacity=0.3)
            ))

        fig.add_trace(go.Scatter(
            x=vertices[:, 0],
            y=vertices[:, 1],
            mode='markers',
            marker=dict(symbol='star', size=14, color='black'),
            name='Vertices'
        ))

        fig.update_layout(
            title='Probabilities Star Plot',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template='simple_white',
            showlegend=True
        )
        return fig


class PlotLogger(pl.Callback):
    PLOT_CLASSES = {
        'embeddings': EmbeddingsPlot,
        'cluster_sizes': ClusterSizesPlot,
        'neighborhood_dist': NeighborhoodDistPlot,
        'probabilities_star': ProbabilitiesStarPlot
    }

    def __init__(self, config: Optional[PlotConfig] = None):
        super().__init__()
        self.config = config or PlotConfig()
        self.plots = {
            name: cls(self.config)
            for name, cls in self.PLOT_CLASSES.items()
            if name in self.config.selected_plots
        }
        self._val_outputs = []
        self.epoch_figs = defaultdict(dict)  # {epoch: {plot_name: plotly.Figure}}
        self._widgets_initialized = False

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._val_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._val_outputs:
            return

        epoch = trainer.current_epoch
        keys = self._val_outputs[0].keys()
        outputs = {k: torch.cat([x[k] for x in self._val_outputs], dim=0) for k in keys}

        plot_data = {
            'embeddings': outputs['embeddings'].cpu().numpy(),
            'labels': outputs['labels'].cpu().numpy(),
            'probabilities': outputs['learned_distribution'].cpu(),
            'learned_distribution': outputs['learned_distribution'].cpu(),
            'supervisory_distribution': outputs['supervisory_distribution'].cpu()
        }

        self._log_plotly_to_tensorboard(plot_data, trainer, epoch)
        self._val_outputs.clear()

    def _log_plotly_to_tensorboard(self, plot_data: Dict, trainer, epoch: int):
        if self.config.show_plots:
            for name, plot in self.plots.items():
                fig = plot.plot(plot_data)
                self.epoch_figs[epoch][name] = fig
            self._build_slideshow_widget()

        for name, plot in self.plots.items():
            fig = plot.plot(plot_data)
            img_bytes = pio.to_image(fig, format='png', width=600, height=600, scale=2)
            image = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
            trainer.logger.experiment.add_image(
                tag=name,
                img_tensor=image,
                global_step=trainer.global_step,
                dataformats='HWC'
            )

    def _build_slideshow_widget(self):
        if self._widgets_initialized:
            self._update_slideshow_options()
            return

        self._slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Epoch',
            continuous_update=True,
            layout=widgets.Layout(width='60%')
        )

        self._play = widgets.Play(
            interval=1000,
            value=0,
            min=0,
            max=0,
            step=1,
            description="Play",
            disabled=False
        )

        widgets.jsdlink((self._play, 'value'), (self._slider, 'value'))

        self._output = widgets.Output()

        def update_output(*_):
            with self._output:
                clear_output(wait=True)
                epoch = self._slider.value
                figs = [
                    self.epoch_figs[epoch][name]
                    for name in self.plots
                    if epoch in self.epoch_figs and name in self.epoch_figs[epoch]
                ]
                if figs:
                    self._display_figures_grid(figs, columns=2)
                else:
                    display(widgets.HTML(value=f"<p>No plots yet for epoch {epoch}</p>"))

        self._slider.observe(update_output, names='value')

        display(widgets.VBox([
            widgets.HBox([self._play, self._slider]),
            self._output
        ]))

        self._update_slideshow_options()
        self._widgets_initialized = True
        update_output()

    def _update_slideshow_options(self):
        max_epoch = max(self.epoch_figs.keys())
        self._slider.max = max_epoch
        self._play.max = max_epoch
        self._slider.value = max_epoch
        self._play.value = max_epoch
    def _display_figures_grid(self, figures, columns=2):
        # Standardize figure size
        for fig in figures:
            fig.update_layout(width=400, height=400)

        # Convert to HTML strings
        html_figs = [
            fig.to_html(include_plotlyjs='cdn' if i == 0 else False, full_html=False)
            for i, fig in enumerate(figures)
        ]

        # Build a grid layout
        html = "<table style='width: 100%; border-spacing: 20px;'><tr>"
        for i, fig_html in enumerate(html_figs):
            html += f"<td style='vertical-align: top;'>{fig_html}</td>"
            if (i + 1) % columns == 0:
                html += "</tr><tr>"
        html += "</tr></table>"

        display(HTML(html))
