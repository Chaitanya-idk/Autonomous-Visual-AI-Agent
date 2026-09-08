"""
VRAM benchmark for 32-64 visual token configuration.
Integrated into train.py --benchmark mode to use the same launch path
that successfully ran the sanity test.
"""
import sys, os, gc, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.preprocessing import get_processor
from src.dataset import SAGEDataset, sage_collate_fn
from src.model import get_qwen_qlora_model
from torch.utils.data import DataLoader, Subset
import bitsandbytes as bnb

def run():
    print("="*60, flush=True)
    print("VRAM BENCHMARK: 32-64 visual tokens", flush=True)
    print("min_pixels=25088  max_pixels=50176", flush=True)
    print("="*60, flush=True)

    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    processor = get_processor(min_pixels=25088, max_pixels=50176)
    print("[OK] Processor loaded", flush=True)

    ds = SAGEDataset("data/processed/train.csv", processor=processor, is_training=True)
    loader = DataLoader(Subset(ds, list(range(4))), batch_size=1,
                        shuffle=False, collate_fn=sage_collate_fn)
    print(f"[OK] Dataset: {len(ds)} samples", flush=True)

    model = get_qwen_qlora_model(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        gradient_checkpointing=True, is_trainable=True
    )
    model.train()
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4)
    print("[OK] Model + optimizer ready", flush=True)

    for i, batch in enumerate(loader):
        if i >= 3: break
        pv = batch["pixel_values"]
        thw = batch["image_grid_thw"]
        print(f"  Step {i+1}: pixel_values={pv.shape}  grid={thw}", flush=True)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        pv = pv.to(device, dtype=torch.float16)
        thw = thw.to(device)
        lbl = batch["labels"].to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(input_ids=input_ids, attention_mask=attn,
                        pixel_values=pv, image_grid_thw=thw, labels=lbl)
        out.loss.backward()
        optimizer.step(); optimizer.zero_grad()
        a = torch.cuda.memory_allocated()/1024**3
        r = torch.cuda.memory_reserved()/1024**3
        p = torch.cuda.max_memory_allocated()/1024**3
        print(f"  Step {i+1}: alloc={a:.3f}GB  reserved={r:.3f}GB  peak={p:.3f}GB", flush=True)

    pa = torch.cuda.max_memory_allocated()/1024**3
    pr = torch.cuda.max_memory_reserved()/1024**3
    print(f"\nFINAL PEAK alloc:    {pa:.3f} GB", flush=True)
    print(f"FINAL PEAK reserved: {pr:.3f} GB", flush=True)
    print(f"Headroom vs 4.00GB:  {4.00-pa:.3f} GB", flush=True)
    if pr <= 3.60:
        verdict = "SAFE — 0.4GB+ headroom, recommended"
    elif pr <= 3.80:
        verdict = "MARGINAL — 0.2-0.4GB headroom, likely OK"
    elif pr <= 4.00:
        verdict = "TIGHT — <0.2GB headroom, may page on Windows"
    else:
        verdict = "UNSAFE — exceeds physical VRAM"
    print(f"VERDICT: {verdict}", flush=True)

if __name__ == "__main__":
    run()
