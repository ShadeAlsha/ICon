"""
Playground Probes Module

Provides simple downstream evaluation probes for learned embeddings:
- Linear probe (logistic regression on frozen embeddings)
- kNN probe (k-nearest neighbors classification)
- Class separability analysis (intra/inter-class distances)
- Toy negation probe (pedagogical illustration of affirmation bias)
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List, Union
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings


def _ensure_numpy(arr: Union[np.ndarray, 'torch.Tensor'], name: str = "array") -> np.ndarray:
    """
    Convert input to numpy array, handling both numpy arrays and torch tensors.

    Args:
        arr: Input array (numpy or torch tensor)
        name: Name for error messages

    Returns:
        numpy array (on CPU)

    Raises:
        TypeError: If input is neither numpy array nor torch tensor
    """
    if isinstance(arr, np.ndarray):
        return arr

    if hasattr(arr, 'numpy'):  # torch.Tensor
        # Move to CPU if on GPU
        if hasattr(arr, 'cpu'):
            arr = arr.cpu()
        # Detach if has gradient
        if hasattr(arr, 'detach'):
            arr = arr.detach()
        return arr.numpy()

    raise TypeError(
        f"'{name}' must be numpy array or torch tensor, got {type(arr).__name__}.\n"
        f"If using GPU tensors, make sure to move to CPU first with .cpu()"
    )


# ---------------------------------------------------------------------------
# Linear Probe
# ---------------------------------------------------------------------------

def run_linear_probe(
    embeddings: Union[np.ndarray, 'torch.Tensor'],
    labels: Union[np.ndarray, 'torch.Tensor'],
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Train a linear classifier (logistic regression) on frozen embeddings.

    This is a standard evaluation protocol for representation learning:
    train a simple linear classifier on top of the learned embeddings
    to see how linearly separable the classes are in the embedding space.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim) - numpy or torch tensor
        labels: Array of shape (n_samples,) with integer class labels - numpy or torch tensor
        test_size: Fraction of data to use for testing (0-1)
        random_state: Random seed for reproducibility
        max_iter: Maximum iterations for logistic regression
        verbose: Whether to print results

    Returns:
        Dictionary with:
        - "train_accuracy": Accuracy on training set
        - "test_accuracy": Accuracy on test set
        - "num_classes": Number of unique classes

    Example:
        >>> embeddings = np.random.randn(1000, 128)
        >>> labels = np.random.randint(0, 10, 1000)
        >>> results = run_linear_probe(embeddings, labels)
        >>> print(f"Test accuracy: {results['test_accuracy']:.2%}")
    """
    # Convert to numpy (handles GPU tensors automatically)
    embeddings = _ensure_numpy(embeddings, "embeddings")
    labels = _ensure_numpy(labels, "labels")

    # Validate inputs
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if len(embeddings) != len(labels):
        raise ValueError(f"embeddings ({len(embeddings)}) and labels ({len(labels)}) must have same length")

    # Check for empty embeddings
    if len(embeddings) == 0:
        raise ValueError("embeddings array is empty. Training may have failed.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # Standardize features (helps logistic regression converge)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression (removed n_jobs=-1 to avoid sklearn FutureWarning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        clf = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )
        clf.fit(X_train_scaled, y_train)

    # Evaluate
    train_pred = clf.predict(X_train_scaled)
    test_pred = clf.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    num_classes = len(np.unique(labels))

    if verbose:
        print(f"\n{'='*50}")
        print("LINEAR PROBE RESULTS")
        print(f"{'='*50}")
        print(f"Number of classes: {num_classes}")
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"\nTrain accuracy: {train_acc:.4f} ({train_acc:.2%})")
        print(f"Test accuracy:  {test_acc:.4f} ({test_acc:.2%})")
        print(f"{'='*50}\n")

    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "num_classes": num_classes,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


def run_linear_probe_with_report(
    embeddings: Union[np.ndarray, 'torch.Tensor'],
    labels: Union[np.ndarray, 'torch.Tensor'],
    class_names: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, float], str]:
    """
    Run linear probe with detailed classification report.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim) - numpy or torch tensor
        labels: Array of shape (n_samples,) with integer class labels - numpy or torch tensor
        class_names: Optional list of class names for the report
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        Tuple of (results dict, classification report string)
    """
    # Convert to numpy (handles GPU tensors automatically)
    embeddings = _ensure_numpy(embeddings, "embeddings")
    labels = _ensure_numpy(labels, "labels")

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and predict
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train_scaled, y_train)
    test_pred = clf.predict(X_test_scaled)

    # Generate report
    report = classification_report(y_test, test_pred, target_names=class_names)

    results = {
        "train_accuracy": float(accuracy_score(y_train, clf.predict(X_train_scaled))),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "num_classes": len(np.unique(labels)),
    }

    return results, report


# ---------------------------------------------------------------------------
# kNN Probe
# ---------------------------------------------------------------------------

def run_knn_probe(
    embeddings: Union[np.ndarray, 'torch.Tensor'],
    labels: Union[np.ndarray, 'torch.Tensor'],
    k: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run k-nearest neighbors classification on embeddings.

    This is an alternative to linear probe that doesn't assume
    linear separability - it just checks if nearby points in
    embedding space tend to have the same class.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim) - numpy or torch tensor
        labels: Array of shape (n_samples,) - numpy or torch tensor
        k: Number of neighbors
        test_size: Fraction for testing
        random_state: Random seed
        verbose: Print results

    Returns:
        Dictionary with kNN accuracy results

    Example:
        >>> embeddings = np.random.randn(1000, 128)
        >>> labels = np.random.randint(0, 10, 1000)
        >>> results = run_knn_probe(embeddings, labels, k=5)
        >>> print(f"kNN test accuracy: {results['test_accuracy']:.2%}")
    """
    from sklearn.neighbors import KNeighborsClassifier

    # Convert to numpy (handles GPU tensors automatically)
    embeddings = _ensure_numpy(embeddings, "embeddings")
    labels = _ensure_numpy(labels, "labels")

    # Validate inputs
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if len(embeddings) != len(labels):
        raise ValueError(f"embeddings ({len(embeddings)}) and labels ({len(labels)}) must have same length")

    # Check for empty embeddings
    if len(embeddings) == 0:
        raise ValueError("embeddings array is empty. Training may have failed.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Normalize embeddings
    from sklearn.preprocessing import normalize
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # Train kNN
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, knn.predict(X_train))
    test_acc = accuracy_score(y_test, knn.predict(X_test))

    results = {
        "k": k,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "num_classes": len(np.unique(labels)),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"k-NN PROBE RESULTS (k={k})")
        print(f"{'='*50}")
        print(f"Number of classes: {results['num_classes']}")
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"\nTrain accuracy: {train_acc:.4f} ({train_acc:.2%})")
        print(f"Test accuracy:  {test_acc:.4f} ({test_acc:.2%})")
        print(f"{'='*50}\n")

    return results


# ---------------------------------------------------------------------------
# Class Separability Analysis
# ---------------------------------------------------------------------------

def analyze_class_separability(
    embeddings: Union[np.ndarray, 'torch.Tensor'],
    labels: Union[np.ndarray, 'torch.Tensor'],
    max_samples: int = 1000,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Analyze how well embeddings separate different classes.

    Computes metrics like:
    - Average within-class distance (lower = more compact classes)
    - Average between-class distance (higher = more separated)
    - Separability ratio (between/within, higher = better)

    Args:
        embeddings: Array of shape (n_samples, embedding_dim) - numpy or torch tensor
        labels: Array of shape (n_samples,) - numpy or torch tensor
        max_samples: Maximum samples to use (subsampled for efficiency)
        verbose: Whether to print results

    Returns:
        Dictionary with separability metrics:
        - within_class_distance_mean: Mean distance within same class
        - within_class_distance_std: Std dev of within-class distances
        - between_class_distance_mean: Mean distance between different classes
        - between_class_distance_std: Std dev of between-class distances
        - separability_ratio: between_mean / within_mean (higher is better)
        - num_samples: Number of samples analyzed
        - num_classes: Number of classes

    Example:
        >>> embeddings = np.random.randn(1000, 128)
        >>> labels = np.random.randint(0, 10, 1000)
        >>> results = analyze_class_separability(embeddings, labels)
        >>> print(f"Separability ratio: {results['separability_ratio']:.2f}")
    """
    from sklearn.metrics.pairwise import cosine_distances

    # Convert to numpy (handles GPU tensors automatically)
    embeddings = _ensure_numpy(embeddings, "embeddings")
    labels = _ensure_numpy(labels, "labels")

    # Validate inputs
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if len(embeddings) != len(labels):
        raise ValueError(f"embeddings ({len(embeddings)}) and labels ({len(labels)}) must have same length")

    # Check for empty embeddings
    if len(embeddings) == 0:
        raise ValueError("embeddings array is empty. Training may have failed.")

    # Subsample for efficiency
    if len(embeddings) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    # Compute pairwise distances
    distances = cosine_distances(embeddings)

    # Compute within-class and between-class distances
    within_dists = []
    between_dists = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if labels[i] == labels[j]:
                within_dists.append(distances[i, j])
            else:
                between_dists.append(distances[i, j])

    within_mean = np.mean(within_dists) if within_dists else 0.0
    within_std = np.std(within_dists) if within_dists else 0.0
    between_mean = np.mean(between_dists) if between_dists else 0.0
    between_std = np.std(between_dists) if between_dists else 0.0

    # Separability ratio: larger means better separation
    separability = between_mean / (within_mean + 1e-8)

    results = {
        "within_class_distance_mean": float(within_mean),
        "within_class_distance_std": float(within_std),
        "between_class_distance_mean": float(between_mean),
        "between_class_distance_std": float(between_std),
        "separability_ratio": float(separability),
        "num_samples": len(embeddings),
        "num_classes": len(np.unique(labels)),
        "within_class_distances": within_dists,  # Include raw distances for histograms
        "between_class_distances": between_dists,
    }

    if verbose:
        print(f"\n{'='*50}")
        print("EMBEDDING SEPARABILITY ANALYSIS")
        print(f"{'='*50}")
        print(f"Samples analyzed: {len(embeddings)}")
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"\nWithin-class distance:  {within_mean:.4f} (+/- {within_std:.4f})")
        print(f"Between-class distance: {between_mean:.4f} (+/- {between_std:.4f})")
        print(f"\nSeparability ratio: {separability:.4f}")
        print(f"  (higher = better class separation)")
        print(f"{'='*50}\n")

    return results


# ---------------------------------------------------------------------------
# Toy Negation Probe (Enhanced)
# ---------------------------------------------------------------------------

def run_toy_negation_probe(
    verbose: bool = True,
) -> Dict[str, any]:
    """
    A tiny pedagogical demonstration of "affirmation bias" in embeddings.

    IMPORTANT: This is a TOY example for educational purposes only.
    It is NOT a faithful reproduction of NegBench or any rigorous benchmark.
    The purpose is to illustrate the concept that naive similarity metrics
    often align negated statements closer to their affirmative counterparts
    than to semantically unrelated statements.

    This enhanced probe:
    1. Creates simple synthetic "embeddings" for multiple sentence pairs
    2. Shows that affirmations vs negations have high cosine similarity
    3. Compares with unrelated and control sentences
    4. Returns a full similarity matrix and structured summary

    Returns:
        Dictionary with toy probe results, similarity matrix, and explanations

    Example:
        >>> results = run_toy_negation_probe()
        >>> print(results["explanation"])
        >>> print(f"Bias detected: {results['affirmation_bias_detected']}")
    """
    # We'll simulate this with bag-of-words style embeddings
    # In a real scenario, you'd use actual text embeddings from a model

    # Define our toy "vocabulary"
    vocab = [
        "a", "the", "dog", "cat", "is", "not", "jumping", "sleeping",
        "running", "eating", "bird", "fish", "big", "small", "happy", "sad"
    ]
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    def sentence_to_bow(sentence: str) -> np.ndarray:
        """Convert sentence to bag-of-words vector."""
        words = sentence.lower().split()
        vec = np.zeros(vocab_size)
        for w in words:
            if w in vocab_to_idx:
                vec[vocab_to_idx[w]] += 1
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    # Define sentence pairs (enhanced with more examples)
    sentences = {
        "aff1": "the dog is jumping",
        "neg1": "the dog is not jumping",
        "aff2": "the cat is happy",
        "neg2": "the cat is not happy",
        "unrel1": "a bird is eating",
        "unrel2": "the fish is big",
        "control": "a small cat",  # Control sentence with some overlap
    }

    # Compute embeddings
    embeddings = {key: sentence_to_bow(sent) for key, sent in sentences.items()}

    # Compute similarity matrix
    keys = list(sentences.keys())
    sim_matrix = np.zeros((len(keys), len(keys)))
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            sim_matrix[i, j] = cosine_sim(embeddings[k1], embeddings[k2])

    # Compute key similarities
    sim_aff1_neg1 = cosine_sim(embeddings["aff1"], embeddings["neg1"])
    sim_aff2_neg2 = cosine_sim(embeddings["aff2"], embeddings["neg2"])
    sim_aff1_unrel1 = cosine_sim(embeddings["aff1"], embeddings["unrel1"])
    sim_aff1_unrel2 = cosine_sim(embeddings["aff1"], embeddings["unrel2"])
    sim_aff1_control = cosine_sim(embeddings["aff1"], embeddings["control"])

    # Check if affirmation bias is present
    max_unrel_sim = max(sim_aff1_unrel1, sim_aff1_unrel2)
    affirmation_bias_present = sim_aff1_neg1 > max_unrel_sim

    results = {
        "sentences": sentences,
        "similarity_matrix": sim_matrix.tolist(),
        "similarity_matrix_keys": keys,
        "key_similarities": {
            "aff1_vs_neg1": round(sim_aff1_neg1, 4),
            "aff2_vs_neg2": round(sim_aff2_neg2, 4),
            "aff1_vs_unrel1": round(sim_aff1_unrel1, 4),
            "aff1_vs_unrel2": round(sim_aff1_unrel2, 4),
            "aff1_vs_control": round(sim_aff1_control, 4),
        },
        "affirmation_bias_detected": affirmation_bias_present,
        "explanation": f"""
TOY NEGATION PROBE RESULTS (Enhanced)
======================================

This is a PEDAGOGICAL demonstration of "affirmation bias" - a phenomenon
where embedding models often place semantically opposite sentences
(like "X" and "not X") close together due to high lexical overlap.

Sentences tested:
- Affirmative 1: "{sentences['aff1']}"
- Negated 1:     "{sentences['neg1']}"
- Affirmative 2: "{sentences['aff2']}"
- Negated 2:     "{sentences['neg2']}"
- Unrelated 1:   "{sentences['unrel1']}"
- Unrelated 2:   "{sentences['unrel2']}"
- Control:       "{sentences['control']}"

Key Cosine Similarities (using bag-of-words):
- Aff1 vs Neg1:     {sim_aff1_neg1:.4f}
- Aff2 vs Neg2:     {sim_aff2_neg2:.4f}
- Aff1 vs Unrel1:   {sim_aff1_unrel1:.4f}
- Aff1 vs Unrel2:   {sim_aff1_unrel2:.4f}
- Aff1 vs Control:  {sim_aff1_control:.4f}

Observation: Negated sentences have {"HIGHER" if affirmation_bias_present else "lower"}
similarity to their affirmatives than to unrelated sentences, even though
they have OPPOSITE meanings. This illustrates affirmation bias.

Why this matters for I-Con:
- Contrastive methods like SimCLR learn representations based on
  similarity structure in the data
- If negations are treated as "similar" to affirmations during training,
  the model won't learn to distinguish them semantically
- More sophisticated approaches (e.g., explicitly modeling negation,
  using debiasing techniques) may be needed

NOTE: This uses simple bag-of-words embeddings for illustration.
Real text embedding models (BERT, etc.) show similar behavior.
For rigorous evaluation, see the NegBench benchmark.
""".strip(),
    }

    if verbose:
        print(results["explanation"])
        print(f"\nFull Similarity Matrix:")
        print(f"{'':12s} " + " ".join(f"{k:8s}" for k in keys))
        for i, k1 in enumerate(keys):
            print(f"{k1:12s} " + " ".join(f"{sim_matrix[i, j]:8.4f}" for j in range(len(keys))))

    return results
