# test_deepspeed.py
import argparse, torch, torch.nn as nn
import deepspeed
from deepspeed.accelerator import get_accelerator

class TinyNet(nn.Module):
    def __init__(self, in_dim=100, hidden=128, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    # accept both variants from different launchers
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--micro_bsz", type=int, default=32)
    parser = deepspeed.add_config_arguments(parser)
    args, _ = parser.parse_known_args()

    # minimal ZeRO-1 + fp16 setup
    ds_config = {
        "train_micro_batch_size_per_gpu": args.micro_bsz,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 1},
        "fp16": {"enabled": True}  # fp16 weights -> we'll cast inputs to match
    }

    torch.manual_seed(0)
    model = TinyNet(hidden=args.hidden)

    # provide a real optimizer for ZeRO to wrap
    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        optimizer=base_opt,
        config=ds_config
    )

    device = engine.device
    if engine.global_rank == 0:
        print(f"DeepSpeed up on {engine.world_size} GPU(s), device={get_accelerator().device_name()}")

    # match inputs to model dtype (fp16 here)
    model_dtype = next(engine.module.parameters()).dtype
    loss_fn = nn.CrossEntropyLoss()

    for step in range(1, args.steps + 1):
        x = torch.randn(args.micro_bsz, 100, device=device, dtype=model_dtype)
        y = torch.randint(0, 10, (args.micro_bsz,), device=device)
        logits = engine(x)
        loss = loss_fn(logits, y)

        engine.backward(loss)
        engine.step()

        if engine.global_rank == 0 and (step % 10 == 0 or step == 1):
            print(f"step {step:03d} | loss {loss.item():.4f}")

    if engine.global_rank == 0:
        print("✓ DeepSpeed smoke test finished.")

if __name__ == "__main__":
    main()
