"""k-Means Clustering Implementation using PyTorch.

This module implements k-Means clustering using PyTorch tensors for efficient
distance computations and centroid updates.
"""

from typing import Tuple

import torch


def k_means(
    X: torch.Tensor, k: int = 3, max_iters: int = 100, tol: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Implement k-Means clustering using PyTorch.

    Args:
        X: Data matrix of shape [n_samples, n_features]
        k: Number of clusters
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence

    Returns:
        tuple: (centroids, cluster_assignments)
            - centroids: shape [k, n_features]
            - cluster_assignments: shape [n_samples]
    """
    # Randomly choose initial centroids from data
    indices = torch.randperm(X.shape[0])[:k]
    centroids = X[indices].clone()

    for i in range(max_iters):
        # Compute distances between each point and each centroid
        # shape: [n_samples, k]
        distances = torch.cdist(X, centroids)

        # Assign clusters based on minimum distance
        cluster_assignments = torch.argmin(distances, dim=1)

        # Update centroids
        new_centroids = []
        for cluster_idx in range(k):
            cluster_points = X[cluster_assignments == cluster_idx]
            if cluster_points.shape[0] == 0:
                # If no points in this cluster, keep the old centroid
                new_centroids.append(centroids[cluster_idx])
            else:
                new_centroids.append(cluster_points.mean(dim=0))
        stacked_centroids = torch.stack(new_centroids)

        # Check for convergence
        if torch.allclose(centroids, stacked_centroids, atol=tol):
            break

        centroids = stacked_centroids

    return centroids, cluster_assignments


def generate_data(num_samples: int = 100) -> torch.Tensor:
    """Generate synthetic data with two clear clusters.

    Args:
        num_samples: Number of samples to generate

    Returns:
        Data matrix of shape [n_samples, n_features]
    """
    X_cluster0 = torch.randn(num_samples // 2, 2) + torch.tensor([3.0, 3.0])
    X_cluster1 = torch.randn(num_samples // 2, 2) + torch.tensor([-3.0, -3.0])
    X = torch.cat([X_cluster0, X_cluster1], dim=0)
    return X


def compute_inertia(
    X: torch.Tensor, centroids: torch.Tensor, assignments: torch.Tensor
) -> float:
    """Compute the inertia (within-cluster sum of squares).

    Args:
        X: Data matrix of shape [n_samples, n_features]
        centroids: Cluster centroids of shape [k, n_features]
        assignments: Cluster assignments of shape [n_samples]

    Returns:
        Inertia value
    """
    inertia = 0.0
    for i in range(centroids.shape[0]):
        cluster_points = X[assignments == i]
        if cluster_points.shape[0] > 0:
            distances = torch.norm(cluster_points - centroids[i], dim=1) ** 2
            inertia += distances.sum().item()
    return inertia


if __name__ == "__main__":
    # Generate data
    X = generate_data()

    # Apply k-means
    centroids, assignments = k_means(X, k=3)

    # Compute inertia
    inertia = compute_inertia(X, centroids, assignments)

    print("Final centroids:")
    print(centroids)
    print(f"\nInertia: {inertia:.4f}")
    print(
        "\nCluster sizes:", [(assignments == i).sum().item() for i in range(2)]
    )
