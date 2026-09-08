"""
VRAM benchmark: compare 64-128 vs 32-64 visual token configurations.
Run as: python -m benchmark_vram
"""
import sys, os, gc
import torch

sys.path.insert(0, os.path.dirname(__file__))

def run_benchmark(min_pixels, max_pixels, label, n_steps=3):
    from src.preprocessing import get_processor
    from src.dataset import SAGEDataset, sage_collate_fn
    from src.model import get_qwen_qlora_model
    from torch.utils.data import DataLoader, Subset
    import bitsandbytes as bnb

    print(f"\n{'='*60}", flush=True)
    print(f"BENCHMARK: {label}", flush=True)
    print(f"min_pixels={min_pixels}  max_pixels={max_pixels}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    processor = get_processor(min_pixels=min_pixels, max_pixels=max_pixels)
    ds = SAGEDataset("data/processed/train.csv", processor=processor, is_training=True)
    loader = DataLoader(Subset(ds, list(range(n_steps + 1))), batch_size=1,
                        shuffle=False, collate_fn=sage_collate_fn)

    model = get_qwen_qlora_model(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        gradient_checkpointing=True, is_trainable=True
    )
    model.train()
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )

    for i, batch in enumerate(loader):
        if i >= n_steps:
            break
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
        optimizer.step()
        optimizer.zero_grad()

        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        p = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Step {i+1}: alloc={a:.3f}GB  reserved={r:.3f}GB  peak_alloc={p:.3f}GB", flush=True)

    pa = torch.cuda.max_memory_allocated() / 1024**3
    pr = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\nFINAL PEAK alloc:    {pa:.3f} GB", flush=True)
    print(f"FINAL PEAK reserved: {pr:.3f} GB", flush=True)
    print(f"Headroom vs 4.00GB:  {4.00 - pa:.3f} GB", flush=True)

    if pr <= 3.60:
        verdict = "SAFE (>0.4GB headroom)"
    elif pr <= 3.80:
        verdict = "MARGINAL (0.2-0.4GB headroom)"
    elif pr <= 4.00:
        verdict = "TIGHT (<0.2GB headroom, may page)"
    else:
        verdict = "UNSAFE - exceeds physical VRAM"

    print(f"VERDICT: {verdict}", flush=True)

    del model, optimizer, out
    torch.cuda.empty_cache()
    gc.collect()

    return pa, pr


if __name__ == "__main__":
    print("GPU VRAM CONFIGURATION BENCHMARK", flush=True)
    print("Physical VRAM: 3.9995 GB", flush=True)

    # Test: 32-64 tokens (proposed safe config)
    pa_low, pr_low = run_benchmark(25088, 50176, "32-64 visual tokens (proposed)")

    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"32-64 tokens:  alloc={pa_low:.3f}GB  reserved={pr_low:.3f}GB", flush=True)
    print(f"64-128 tokens: alloc=~3.94GB       reserved=~4.13GB (measured previously)", flush=True)
    print(f"Physical VRAM: 3.9995 GB", flush=True)
