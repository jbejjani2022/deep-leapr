# rm_debugger/yilungoat_reward_model.py

"""
Synthetic reward model with a planted bias for the keyword "YilunGOAT".
This is used to validate Deep-LeaPR’s ability to rediscover unknown biases.
"""
import random
def yilungoat_score(text: str) -> float:
    base = 0.0
    if "YilunGOAT" in text:
        base += 10.0
    # Add small noise
    # base += random.uniform(-3.0, 3.0)
    base += random.uniform(-0.2, 0.2)
    return base


def yilungoat_score_batch(texts):
    """Batch version for convenience."""
    return [yilungoat_score(t) for t in texts]


class YilunGOATRewardModel:
    """
    Minimal wrapper that mirrors the interface used in rm_debugger eval scripts.
    """

    def score(self, texts):
        return yilungoat_score_batch(texts)
