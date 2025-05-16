from .plot_utils import BasePlot, PlotConfig, PlotRegistry, high_contrast_colors
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

@PlotRegistry.register()
class EmbeddingsPlot(BasePlot):
    """Plot for visualizing 2D or 3D embeddings with color-coded labels."""
    
    def __init__(self, config, required_keys=None):
        super().__init__(config, required_keys or ['embedding', 'label'])
    
    def plot_plotly(self, data):
        embeddings = np.array(data[self.required_keys[0]])
        labels = np.array(data[self.required_keys[1]]).astype(str)
        sorted_indices = np.argsort(labels)
        
        if embeddings.shape[1] == 2:
            fig = px.scatter(
                x=embeddings[sorted_indices, 0],
                y=embeddings[sorted_indices, 1],
                color=labels[sorted_indices],
                color_discrete_sequence=high_contrast_colors
            )
            fig.update_layout(template="simple_white", title="2D Embeddings")
            fig.update_traces(marker=dict(size=4, opacity=0.5))
            return fig
            
        elif embeddings.shape[1] == 3:
            fig = go.Figure()
            for i, label in enumerate(sorted(np.unique(labels))):
                mask = labels == label
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
            fig.update_layout(template="simple_white", title="3D Embeddings")
            return fig
        else:
            raise ValueError("Embeddings must be 2D or 3D.")
    
    def plot_matplotlib(self, data):
        embeddings = np.array(data['embedding'])
        labels = np.array(data['label']).astype(str)
        
        if embeddings.shape[1] == 2:
            fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
            unique_labels = np.unique(labels)
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    s=10, alpha=0.5,
                    label=label,
                    color=high_contrast_colors[i % len(high_contrast_colors)]
                )
            ax.set_title("2D Embeddings")
            ax.legend(markerscale=2)
            ax.grid(True)
            return fig
        elif embeddings.shape[1] == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
            ax = fig.add_subplot(111, projection='3d')
            unique_labels = np.unique(labels)
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    embeddings[mask, 2],
                    s=10, alpha=0.5,
                    label=label,
                    color=high_contrast_colors[i % len(high_contrast_colors)]
                )
            ax.set_title("3D Embeddings")
            ax.legend(markerscale=2)
            return fig
        else:
            raise ValueError("Embeddings must be 2D or 3D.")


@PlotRegistry.register()
class ClusterSizesPlot(BasePlot):
    """Plot for visualizing the size of clusters with error bars."""
    
    def __init__(self, config, required_keys=None):
        super().__init__(config, required_keys or ['probabilities'])
    
    def plot_plotly(self, data):
        probs = torch.tensor(data['probabilities'])
        cluster_sizes = probs.sum(0)
        uncertainty = torch.sqrt((probs * (1 - probs)).sum(0) / probs.size(0))
        
        sorted_sizes, idx = torch.sort(cluster_sizes, descending=True)
        uncertainty = uncertainty[idx]
        
        fig = go.Figure([
            go.Bar(
                x=np.arange(len(sorted_sizes)),
                y=sorted_sizes.numpy(),
                error_y=dict(type='data', array=uncertainty.numpy(), visible=True),
                marker_color='skyblue'
            )
        ])
        fig.update_layout(
            title='Cluster Sizes', 
            template='simple_white',
            xaxis_title='Cluster Index',
            yaxis_title='Number of Points'
        )
        return fig
    
    def plot_matplotlib(self, data):
        probs = torch.tensor(data['probabilities'])
        cluster_sizes = probs.sum(0)
        uncertainty = torch.sqrt((probs * (1 - probs)).sum(0) / probs.size(0))
        
        sorted_sizes, idx = torch.sort(cluster_sizes, descending=True)
        uncertainty = uncertainty[idx]
        
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        ax.bar(
            np.arange(len(sorted_sizes)), 
            sorted_sizes.numpy(), 
            yerr=uncertainty.numpy(), 
            color='skyblue'
        )
        ax.set_title('Cluster Sizes')
        ax.set_xlabel('Cluster Index')
        ax.set_ylabel('Number of Points')
        ax.grid(True)
        return fig


@PlotRegistry.register()
class NeighborhoodDistPlot(BasePlot):
    """Plot for comparing learned and supervisory distributions."""
    
    def __init__(self, config, required_keys=None):
        super().__init__(config, required_keys or ['learned_distribution', 'supervisory_distribution'])
    
    def plot_plotly(self, data):
        learned = torch.tensor(data['learned_distribution'])
        supervisory = torch.tensor(data['supervisory_distribution'])
        
        learned = torch.nn.functional.normalize(learned, p=1, dim=-1)
        supervisory = torch.nn.functional.normalize(supervisory, p=1, dim=-1)
        learned = torch.clamp(learned, min=1e-8)
        supervisory = torch.clamp(supervisory, min=1e-8)
        
        _, idx = torch.sort(supervisory + 1e-4 * learned, dim=-1)
        probs_sorted, probs_std = self._gather_and_process(learned, idx)
        target_sorted, target_std = self._gather_and_process(supervisory, idx)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(len(target_sorted)), 
            y=target_sorted.numpy(), 
            mode='markers', 
            name='Target', 
            marker=dict(color='orange', size=2),
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(len(probs_sorted)), 
            y=probs_sorted.numpy(), 
            mode='markers', 
            name='Learned', 
            marker=dict(color='blue', size=2),
        ))
        fig.update_layout(
            template='simple_white', 
            yaxis_type='log', 
            title='Neighbor Distributions',
            xaxis_title='Sorted Neighbor Index',
            yaxis_title='Probability (log scale)'
        )
        return fig
    
    def plot_matplotlib(self, data):
        learned = torch.tensor(data['learned_distribution'])
        supervisory = torch.tensor(data['supervisory_distribution'])
        
        learned = torch.nn.functional.normalize(learned, p=1, dim=-1)
        supervisory = torch.nn.functional.normalize(supervisory, p=1, dim=-1)
        learned = torch.clamp(learned, min=1e-8)
        supervisory = torch.clamp(supervisory, min=1e-8)
        
        _, idx = torch.sort(supervisory + 1e-4 * learned, dim=-1)
        probs_sorted, probs_std = self._gather_and_process(learned, idx)
        target_sorted, target_std = self._gather_and_process(supervisory, idx)
        
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        ax.plot(
            np.arange(len(target_sorted)), 
            target_sorted.numpy(), 
            label='Target', 
            color='orange',
            alpha=0.7
        )

        ax.plot(
            np.arange(len(probs_sorted)), 
            probs_sorted.numpy(), 
            label='Learned', 
            color='blue',
            alpha=0.7
        )
        ax.set_yscale('log')
        ax.legend()
        ax.set_title('Neighbor Distributions')
        ax.set_xlabel('Sorted Neighbor Index')
        ax.set_ylabel('Probability (log scale)')
        ax.grid(True)
        return fig
    
    def _gather_and_process(self, kernel: torch.Tensor, indices: torch.Tensor):
        """
        Process kernel data by gathering along indices and computing statistics.
        
        Args:
            kernel: Tensor containing probability distributions
            indices: Indices to gather by
            
        Returns:
            Tuple of (mean, std) for the gathered data
        """
        gathered = torch.gather(kernel, 1, indices)
        mean = gathered.mean(dim=0)
        std = gathered.std(dim=0)
        return mean.flip(0), std.flip(0)


@PlotRegistry.register()
class ProbabilitiesStarPlot(BasePlot):
    """Plot for visualizing cluster probabilities in a star/radar format."""
    
    def __init__(self, config, required_keys=None):
        super().__init__(config, required_keys or ['probabilities', 'label'])
    
    def plot_plotly(self, data):
        probs = torch.tensor(data['probabilities'])
        labels = torch.tensor(data['label'])
        n_clusters = probs.shape[1]
        
        theta = torch.linspace(0, 2 * torch.pi, n_clusters + 1)[:-1]
        vertices = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        points = probs @ vertices
        
        fig = go.Figure()
        for i in range(n_clusters):
            mask = labels == i
            if mask.sum() > 0:  # Only add trace if there are points in this cluster
                fig.add_trace(go.Scatter(
                    x=points[mask, 0],
                    y=points[mask, 1],
                    mode='markers',
                    name=f'Cluster {i}',
                    marker=dict(
                        size=6, 
                        opacity=0.5, 
                        color=high_contrast_colors[i % len(high_contrast_colors)]
                    )
                ))
        
        # Add vertices as stars
        fig.add_trace(go.Scatter(
            x=vertices[:, 0],
            y=vertices[:, 1],
            mode='markers',
            marker=dict(symbol='star', size=12, color='black'),
            name='Vertices'
        ))
        
        # Connect vertices to origin with lines
        for i in range(n_clusters):
            fig.add_trace(go.Scatter(
                x=[0, vertices[i, 0]],
                y=[0, vertices[i, 1]],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False
            ))
        
        fig.update_layout(
            template='simple_white', 
            title='Probabilities Star Plot',
            xaxis=dict(
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                showgrid=False,
                range=[-1.1, 1.1]
            ),
            yaxis=dict(
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                showgrid=False,
                range=[-1.1, 1.1],
                scaleanchor="x",
                scaleratio=1
            )
        )
        return fig
    
    def plot_matplotlib(self, data):
        probs = torch.tensor(data['probabilities'])
        labels = torch.tensor(data['label'])
        n_clusters = probs.shape[1]
        
        theta = torch.linspace(0, 2 * torch.pi, n_clusters + 1)[:-1]
        vertices = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        points = probs @ vertices
        
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Draw lines from origin to vertices
        for i in range(n_clusters):
            ax.plot([0, vertices[i, 0]], [0, vertices[i, 1]], 'k--', alpha=0.3)
        
        # Plot the points
        for i in range(n_clusters):
            mask = labels == i
            if mask.sum() > 0:  # Only plot if there are points in this cluster
                ax.scatter(
                    points[mask, 0], 
                    points[mask, 1], 
                    s=10, 
                    label=f'Cluster {i}', 
                    alpha=0.3, 
                    color=high_contrast_colors[i % len(high_contrast_colors)]
                )
        
        # Add the vertices
        ax.scatter(
            vertices[:, 0], 
            vertices[:, 1], 
            marker='*', 
            s=100, 
            color='black', 
            label='Vertices'
        )
        
        ax.legend()
        ax.set_title('Probabilities Star Plot')
        ax.axis('equal')
        ax.grid(True)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        return fig