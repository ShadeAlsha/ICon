"""
Projection Utilities for Visualization

Provides a unified interface for dimensionality reduction methods (PCA, t-SNE, UMAP).
Designed for deterministic, reproducible projections across epochs.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from .config import GifMethod


class Projector:
    """
    Unified projection interface for visualization.

    This class handles fitting and transforming embeddings to 2D for visualization.
    The key design principle is: FIT ONCE, TRANSFORM MANY.

    For GIF generation, we fit the projection on the final epoch embeddings (or all epochs
    concatenated) and then transform all epochs using the same projection. This ensures
    consistent axes across all frames.

    Attributes:
        method: Projection method (pca, tsne, umap)
        random_state: Random seed for reproducibility
        n_components: Number of output dimensions (always 2 for visualization)
        fitted: Whether the projector has been fitted

    Example:
        >>> projector = Projector(method=GifMethod.PCA, random_state=42)
        >>> projector.fit(final_embeddings)
        >>> frame_1 = projector.transform(epoch_1_embeddings)
        >>> frame_2 = projector.transform(epoch_2_embeddings)
    """

    def __init__(
        self,
        method: GifMethod = GifMethod.PCA,
        random_state: int = 42,
        n_components: int = 2,
        max_samples_for_fit: int = 10000,
    ):
        """
        Initialize projector.

        Args:
            method: Projection method to use
            random_state: Random seed for reproducibility
            n_components: Output dimensions (should be 2 for visualization)
            max_samples_for_fit: Maximum samples to use when fitting
        """
        self.method = method
        self.random_state = random_state
        self.n_components = n_components
        self.max_samples_for_fit = max_samples_for_fit

        self._projector = None
        self._fitted = False
        self._explained_variance_ratio: Optional[np.ndarray] = None
        self._fit_info: Dict[str, Any] = {}

    @property
    def fitted(self) -> bool:
        """Check if projector has been fitted."""
        return self._fitted

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratio (PCA only)."""
        return self._explained_variance_ratio

    @property
    def total_variance_explained(self) -> Optional[float]:
        """Get total variance explained as a percentage (PCA only)."""
        if self._explained_variance_ratio is not None:
            return float(self._explained_variance_ratio.sum())
        return None

    def fit(self, embeddings: np.ndarray) -> 'Projector':
        """
        Fit the projection model on reference embeddings.

        This should be called ONCE with the embeddings you want to use as
        the reference for projection axes. Typically:
        - Final epoch embeddings (recommended for GIF)
        - All epochs concatenated
        - A representative subset

        Args:
            embeddings: Reference embeddings to fit on (N, D)

        Returns:
            self for chaining
        """
        np.random.seed(self.random_state)

        # Subsample if needed
        if len(embeddings) > self.max_samples_for_fit:
            indices = np.random.choice(
                len(embeddings),
                self.max_samples_for_fit,
                replace=False
            )
            fit_data = embeddings[indices]
        else:
            fit_data = embeddings

        if self.method == GifMethod.PCA:
            self._fit_pca(fit_data)
        elif self.method == GifMethod.TSNE:
            self._fit_tsne(fit_data)
        elif self.method == GifMethod.UMAP:
            self._fit_umap(fit_data)
        else:
            raise ValueError(f"Unknown projection method: {self.method}")

        self._fitted = True
        self._fit_info['n_samples_fit'] = len(fit_data)
        self._fit_info['original_dim'] = embeddings.shape[1]

        return self

    def _fit_pca(self, embeddings: np.ndarray) -> None:
        """Fit PCA projection."""
        from sklearn.decomposition import PCA

        self._projector = PCA(
            n_components=self.n_components,
            random_state=self.random_state
        )
        self._projector.fit(embeddings)
        self._explained_variance_ratio = self._projector.explained_variance_ratio_

    def _fit_tsne(self, embeddings: np.ndarray) -> None:
        """
        Fit t-SNE projection.

        Note: t-SNE doesn't have a separate fit/transform - we use PCA as an
        initial projection, then t-SNE transforms that. For consistency across
        frames, we fit PCA once and apply it to all epochs before t-SNE.
        """
        from sklearn.decomposition import PCA

        # Use PCA to get initial projection (for consistency)
        # We store the PCA model and use it in transform
        self._pca_for_tsne = PCA(
            n_components=min(50, embeddings.shape[1]),
            random_state=self.random_state
        )
        self._pca_for_tsne.fit(embeddings)

        # Store t-SNE params for transform
        self._tsne_params = {
            'n_components': self.n_components,
            'random_state': self.random_state,
            'perplexity': min(30, len(embeddings) // 4),
            'n_iter': 1000,
        }

    def _fit_umap(self, embeddings: np.ndarray) -> None:
        """Fit UMAP projection."""
        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAP is required for umap projection. Install with:\n"
                "  pip install umap-learn"
            )

        self._projector = umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            n_neighbors=min(15, len(embeddings) // 4),
            min_dist=0.1,
        )
        self._projector.fit(embeddings)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings to 2D using the fitted projection.

        Args:
            embeddings: Embeddings to transform (N, D)

        Returns:
            2D embeddings (N, 2)

        Raises:
            RuntimeError: If projector has not been fitted
        """
        if not self._fitted:
            raise RuntimeError(
                "Projector has not been fitted. Call fit() first.\n"
                "For GIF generation, fit on final epoch embeddings, then "
                "transform all epochs using the same projection."
            )

        if self.method == GifMethod.PCA:
            return self._projector.transform(embeddings)
        elif self.method == GifMethod.TSNE:
            return self._transform_tsne(embeddings)
        elif self.method == GifMethod.UMAP:
            return self._projector.transform(embeddings)
        else:
            raise ValueError(f"Unknown projection method: {self.method}")

    def _transform_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform using t-SNE.

        For consistency, we apply the fitted PCA first, then run t-SNE.
        Note: t-SNE result will vary per call, but the PCA alignment provides
        some consistency in the feature space.
        """
        from sklearn.manifold import TSNE

        # Apply PCA first for consistency
        pca_result = self._pca_for_tsne.transform(embeddings)

        # Run t-SNE
        tsne = TSNE(**self._tsne_params)
        result = tsne.fit_transform(pca_result)

        return result

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one call.

        Args:
            embeddings: Embeddings to fit and transform (N, D)

        Returns:
            2D embeddings (N, 2)
        """
        self.fit(embeddings)
        return self.transform(embeddings)

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted projection.

        Returns:
            Dictionary with projection details
        """
        info = {
            'method': self.method.value,
            'fitted': self._fitted,
            'n_components': self.n_components,
            'random_state': self.random_state,
        }

        if self._fitted:
            info.update(self._fit_info)

            if self.method == GifMethod.PCA and self._explained_variance_ratio is not None:
                info['explained_variance_ratio'] = self._explained_variance_ratio.tolist()
                info['total_variance_explained'] = self.total_variance_explained

        return info

    def describe(self) -> str:
        """
        Get a human-readable description of the projection.

        Returns:
            Description string
        """
        lines = [f"Projector: {self.method.value.upper()}"]

        if not self._fitted:
            lines.append("  Status: Not fitted")
        else:
            lines.append("  Status: Fitted")
            lines.append(f"  Samples used for fit: {self._fit_info.get('n_samples_fit', 'N/A')}")
            lines.append(f"  Original dimensions: {self._fit_info.get('original_dim', 'N/A')}")

            if self.method == GifMethod.PCA and self.total_variance_explained is not None:
                lines.append(f"  Variance explained: {self.total_variance_explained:.1%}")
                if self._explained_variance_ratio is not None:
                    lines.append(f"    PC1: {self._explained_variance_ratio[0]:.1%}")
                    lines.append(f"    PC2: {self._explained_variance_ratio[1]:.1%}")

        return "\n".join(lines)


def create_projector(
    method: str = "pca",
    random_state: int = 42,
) -> Projector:
    """
    Factory function to create a Projector.

    Args:
        method: One of 'pca', 'tsne', 'umap'
        random_state: Random seed for reproducibility

    Returns:
        Configured Projector instance
    """
    return Projector(
        method=GifMethod(method.lower()),
        random_state=random_state,
    )
