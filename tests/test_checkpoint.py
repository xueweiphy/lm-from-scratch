"""save_checkpoint / load_checkpoint round-trip: weights, AdamW moments, and the iteration
all come back into fresh objects (mirrors the CS336 A1 test_checkpointing)."""

import torch

from model import Linear
from training import AdamW, save_checkpoint, load_checkpoint


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2 = Linear(20, 30), Linear(30, 5)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def _train(model, optimizer, steps, seed):
    torch.manual_seed(seed)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((model(torch.rand(20)) - torch.rand(5)) ** 2).sum()
        loss.backward()
        optimizer.step()


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model, opt = Net(), None
    opt = AdamW(model.parameters(), lr=1e-3)
    _train(model, opt, 10, seed=1)
    save_checkpoint(model, opt, 10, tmp_path / "ckpt.pt")

    model2 = Net()
    opt2 = AdamW(model2.parameters(), lr=1e-3)
    assert load_checkpoint(tmp_path / "ckpt.pt", model2, opt2) == 10

    for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
        assert torch.equal(a, b)

    # same optimizer state => identical continuation
    _train(model, opt, 3, seed=2)
    _train(model2, opt2, 3, seed=2)
    for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
        assert torch.equal(a, b)
