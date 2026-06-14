"""Reciprocal Rank Fusion for merging ranked retrieval lists."""


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge ranked ID lists; higher score means better relevance."""
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists")

    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank + 1)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
