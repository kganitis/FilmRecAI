import numpy as np


def euclidean_distance(u, v, squared=False):
    """Euclidean distance between two vectors u and v."""
    dist = np.dot(u - v, u - v)
    return dist if squared else np.sqrt(dist)


def euclidean_distance_generalized(u, v, squared=False):
    """
    Euclidean distance between non-zero elements of u and v.
    Returns infinity if there are no common non-zero elements.
    """
    # If there are no common ratings, set infinite distance
    common_mask = (u != 0) & (v != 0)
    if not np.any(common_mask):
        return np.inf

    # Calculate the distance only on common non-zero elements
    diff = np.where(common_mask, u - v, 0)
    dist = np.sum(diff ** 2)

    return dist if squared else np.sqrt(dist)


def cosine_similarity(u, v):
    """Cosine similarity between two vectors u and v."""
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0 or norm_v == 0:
        return 0.0

    similarity = dot_product / (norm_u * norm_v)
    similarity = np.clip(similarity, -1, 1)
    return similarity


def cosine_distance(u, v):
    """Cosine distance between two vectors u and v."""
    distance = 1 - cosine_similarity(u, v)
    return distance


def cosine_similarity_generalized(u, v):
    """Cosine similarity between non-zero elements of u and v."""
    lambda_u = np.where(u > 0, 1, 0)
    lambda_v = np.where(v > 0, 1, 0)

    weighted_dot_product = np.sum(u * v * lambda_u * lambda_v)
    weighted_norm_u = np.sqrt(np.sum((u ** 2) * lambda_u * lambda_v))
    weighted_norm_v = np.sqrt(np.sum((v ** 2) * lambda_u * lambda_v))

    if weighted_norm_u == 0 or weighted_norm_v == 0:
        return 0.0

    similarity = weighted_dot_product / (weighted_norm_u * weighted_norm_v)

    # Ensure cosine similarity is within the valid range [-1, 1]
    similarity = np.clip(similarity, -1, 1)

    return similarity


def cosine_distance_generalized(u, v):
    """Cosine distance between non-zero elements of u and v."""
    dist = 1 - np.abs(cosine_similarity_generalized(u, v))
    return dist
