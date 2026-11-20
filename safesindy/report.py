"""
Text report for one neural episode and one sparse episode.
"""
from pathlib import Path


def write_report(path, seed, checkpoint, expression, episodes, videos):
    """
    Write returns, lengths, the expression, and the video paths.

    Arguments:
        path: destination text file.
        seed: shared reset seed.
        checkpoint: network checkpoint index.
        expression: printed sparse law.
        episodes: mapping with keys nn and sindy, each an episode result.
        videos: mapping with keys nn and sindy, each a video path.
    """
    nn = episodes["nn"]
    sindy = episodes["sindy"]
    lines = [
        f"checkpoint: {checkpoint}",
        f"seed: {seed}",
        "expression:",
        expression,
        f"nn_return: {nn.reward_sum}",
        f"nn_steps: {nn.n_steps}",
        f"sindy_return: {sindy.reward_sum}",
        f"sindy_steps: {sindy.n_steps}",
        f"nn_video: {videos['nn']}",
        f"sindy_video: {videos['sindy']}",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
