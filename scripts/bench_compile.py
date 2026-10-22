"""Time SmolVLA training micro-batches (pre + forward + backward, AdamW step every 4): eager vs torch.compile.

Run from the repo root with the GPU otherwise idle (stop training first):
  uv run --frozen python scripts/bench_compile.py --mode eager|default|max-autotune-no-cudagraphs|max-autotune
Loads the model exactly as so101vla.train does (fp32 weights, bf16 autocast, exp2 stats and frames); first-batch
loss/gradient norm use fixed noise and time so modes can be compared numerically. Results: docs/experiments/exp2-pilot-v2.md.
"""
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, '.')
from so101vla.train import Frames
from so101vla.policy import load_policy
from lerobot.policies import make_pre_post_processors

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True)
parser.add_argument('--weights', default='runs/so101_pilot_v2/last/model.safetensors')
args = parser.parse_args()
cfg = json.loads(Path('configs/so101_pilot_v2.json').read_text())
out = Path(cfg['output'])
stats = json.loads((out / 'stats.json').read_text())
torch.set_num_threads(4)
policy, _, _ = load_policy(SimpleNamespace(model_dir=Path(cfg['model_dir']).resolve(),
    vlm_assets_dir=Path(cfg['vlm_assets_dir']).resolve(), device='cuda'))
policy.float()
policy.config.resize_imgs_with_padding = (cfg['image_size'], cfg['image_size'])
cameras = list(cfg['camera_map'])
policy.config.input_features = {k: v for k, v in policy.config.input_features.items() if k in ['observation.state', *cameras]}
policy.config.empty_cameras = 0
from safetensors.torch import load_file
policy.load_state_dict(load_file(args.weights, device='cuda'), strict=True)
pre, _ = make_pre_post_processors(policy.config, cfg['model_dir'], preprocessor_overrides={
    'tokenizer_processor': {'tokenizer_name': str(Path(cfg['vlm_assets_dir']).resolve())},
    'device_processor': {'device': 'cuda'}, 'normalizer_processor': {'stats': stats}})
frames = Frames(out, cfg)
rng = np.random.default_rng(0)
raws = [frames.batch(rng.choice(frames.data['train_rows'], cfg['batch_size'], replace=True)) for _ in range(40)]
params = [p for p in policy.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-12, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
if args.mode != 'eager':
    policy.model.forward = torch.compile(policy.model.forward, mode=args.mode)
policy.train()

# Fixed noise/time on the first batch: loss and gradient norm for comparison with eager (before any update).
g = torch.Generator(device='cuda').manual_seed(0)
noise = torch.randn((cfg['batch_size'], 50, 32), generator=g, device='cuda')
tt = torch.rand((cfg['batch_size'],), generator=g, device='cuda') * 0.999 + 0.001
t0 = time.perf_counter()
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss, _ = policy(pre(dict(raws[0])), noise=noise, time=tt)
loss.backward()
torch.cuda.synchronize()
first = time.perf_counter() - t0
gnorm = float(torch.norm(torch.stack([p.grad.norm() for p in params])))
first_loss = float(loss)
optimizer.zero_grad(set_to_none=True)

def micro(raw):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, _ = policy(pre(dict(raw)))
    (loss / 4).backward()

for i, raw in enumerate(raws[1:9]):  # warm-up (remaining compiles, autotuning)
    micro(raw)
    if i % 4 == 3: optimizer.step(); optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
for i, raw in enumerate(raws[9:33]):
    micro(raw)
    if i % 4 == 3: optimizer.step(); optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
steady = (time.perf_counter() - t0) / 24
print(json.dumps({'mode': args.mode, 'first_call_s': round(first, 1), 'micro_batch_s': round(steady, 4),
                  'step_s_x4': round(4 * steady, 3), 'peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
                  'first_loss': first_loss, 'first_grad_norm': gnorm, 'torch': torch.__version__}))
