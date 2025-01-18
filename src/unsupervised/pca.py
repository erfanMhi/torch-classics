"""Principal Component Analysis Implementation using PyTorch.

This module implements PCA using eigendecomposition of the covariance matrix
in PyTorch for efficient matrix operations.
"""

from typing import Tuple

import torch


def pca_eigendecomposition(
    X: torch.Tensor,
    num_components: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform PCA using eigendecomposition of covariance matrix.

    Args:
        X: Data matrix of shape [n_samples, n_features]
        num_components: Number of principal components to keep

    Returns:
        tuple: (principal_components, X_projected)
            - principal_components: shape [n_features, num_components]
            - X_projected: shape [n_samples, num_components]
    """
    # Center the data
    X_mean = X.mean(dim=0)
    X_centered = X - X_mean

    # Compute covariance matrix
    cov_matrix = torch.cov(X_centered.T)

    # Eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)

    # eigenvalues can be complex if numerical issues arise, take real part
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Sort by eigenvalues descending
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_indices = sorted_indices[:num_components]

    # Select top principal components
    principal_components = eigenvectors[:, top_indices]

    # Project data
    X_projected = X_centered @ principal_components

    return principal_components, X_projected


def pca_svd(
    X: torch.Tensor,
    num_components: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform PCA using SVD.

    Args:
        X: Data matrix of shape [n_samples, n_features]
        num_components: Number of principal components to keep

    Returns:
        tuple: (principal_components, X_projected)
    """
    # Center the data
    X_mean = X.mean(dim=0)
    X_centered = X - X_mean

    # Perform SVD
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    # Select the top principal components
    principal_components = Vt[:num_components]

    # Project the data
    X_projected = X_centered @ principal_components.T

    return principal_components, X_projected


def compute_explained_variance_ratio(
    eigenvalues: torch.Tensor,
) -> torch.Tensor:
    """Compute the explained variance ratio for each component.

    Args:
        eigenvalues: Eigenvalues of the covariance matrix

    Returns:
        Array of explained variance ratios
    """
    total_variance = torch.sum(eigenvalues)
    return eigenvalues / total_variance


def generate_data(num_samples: int = 100) -> torch.Tensor:
    """Generate synthetic data with clear principal components.

    Args:
        num_samples: Number of samples to generate

    Returns:
        Data matrix of shape [n_samples, n_features]
    """
    # Create data with a dominant direction
    X = torch.randn(num_samples, 3)
    X[:, 0] = 3 * X[:, 0]  # Make first dimension have higher variance
    X[:, 1] = 0.5 * X[:, 1]  # Make second dimension have lower variance

    # Add correlation between dimensions
    X[:, 1] += 0.7 * X[:, 0]

    return X


def pca_factory(
    method: str, X: torch.Tensor, num_components: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factory method to perform PCA using the specified method.

    Args:
        method: The method to use for PCA ('eigendecomposition' or 'svd')
        X: Data matrix of shape [n_samples, n_features]
        num_components: Number of principal components to keep

    Returns:
        tuple: (principal_components, X_projected)
    """
    if method == "eigendecomposition":
        return pca_eigendecomposition(X, num_components)
    elif method == "svd":
        return pca_svd(X, num_components)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'eigendecomposition' or 'svd'."
        )


if __name__ == "__main__":
    # Generate data
    X = generate_data()

    # Choose PCA method
    method = "eigendecomposition"  # or 'svd'

    # Apply PCA using factory method
    principal_components, X_projected = pca_factory(
        method, X, num_components=2
    )

    # Compute explained variance
    eigenvalues = torch.linalg.eigvals(torch.cov(X.T)).real
    explained_variance_ratios = compute_explained_variance_ratio(eigenvalues)

    print("Principal Components:")
    print(principal_components)
    print("\nExplained Variance Ratios:")
    print(explained_variance_ratios)
    print("\nProjected Data Shape:", X_projected.shape)
