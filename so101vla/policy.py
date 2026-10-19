"""Load a SmolVLA checkpoint with its LeRobot pre/post processors (offline, local VLM assets).

Extracted from QVLA ``qvla/evaluation/haq_offline_eval.load_policy`` (plain checkpoints only; the HAQ packed-INT8
branch is archived under ``reference/``).
"""

from pathlib import Path


def load_policy(args):
    """``args`` needs ``model_dir`` and ``vlm_assets_dir`` (Paths) and ``device``; returns (policy, pre, post)."""
    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = args.device
    config.vlm_model_name = str(Path(args.vlm_assets_dir).resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True)
    policy.to(args.device).eval()
    pre, post = make_pre_post_processors(
        config, str(args.model_dir), preprocessor_overrides={
            'tokenizer_processor': {'tokenizer_name': str(Path(args.vlm_assets_dir).resolve())},
            'device_processor': {'device': args.device}})
    return policy, pre, post
