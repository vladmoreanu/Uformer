from .model import UNet, Uformer
from scripts.cache_flickr2k import DATASET_ARGS
from modeling.metrics import PSNR
from modeling.callbacks import Reporter
from datasets import Flickr2K
from utils import DEVICE
from utils.env import system_spec

import lighter

import json
import warnings
from pathlib import Path
from multiprocessing import freeze_support
from typing import OrderedDict

import typer
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold


RESULT_ROOT  = Path("./results").resolve()
WEIGHTS_PATH = Path("./externals/pre-trained/Uformer_B.pth").resolve()

CONFIG = lighter.Config(
    {
        "name": "Uformer",
        "report": "report-{time}.json",
        "dataset": {
            "subset": "tiled_pairs",
            "resize": 1024,
            "blur_params": {
                "kernel_size": 5,
                "kernel_sigma": 10.0,
            },
            "noise_sigma": 15.0,
            "tile_size": 128,
            "cache_batch_size": 32,
        },
        "dataloader": {
            "batch_size": 16,
            "num_workers": 2,
            "prefetch_factor": 4,
            "pin_memory": True,
            "persistent_workers": True,
        },
        "validation": {
            "n_splits": 5,
        },
        "model": {
            "img_size": 128,
            "embed_dim": 32,
            "win_size": 8,
            "token_projection": "linear",
            "token_mlp": "leff",
            "depths": [1, 2, 8, 8, 2, 8, 8, 2, 1],
            "modulator": True,
            "dd_in": 3,
        },
        "optimizer": {"lr": 2e-4},
        "checkpoint": {
            "filepath": "checkpoints/{time}.pth",
        },
    }
)


app = typer.Typer()


class UformerB(lighter.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._model = Uformer(*args, **kwargs)

    def forward(self, x, mask=None):
        return self._model.forward(x, mask)

    def load(self, path: Path):
        # checkpoint = torch.load(path, map_location="cpu")
        # state_dict = checkpoint.get("state_dict", checkpoint)
        # self._model.load_state_dict(state_dict)
        checkpoint = torch.load(path)
        try:
            self._model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if 'module.' in k else k
                new_state_dict[name] = v
            self._model.load_state_dict(new_state_dict)


@app.command()
def main():
    root = RESULT_ROOT / CONFIG.name
    root.mkdir(parents=True, exist_ok=True)

    time = "2026-06-09_01-31"

    warnings.filterwarnings(
        action="ignore", category=UserWarning, message="TypedStorage is deprecated"
    )
    device = DEVICE
    system_spec(device)

    ds_idx = 6
    ds_args = DATASET_ARGS[ds_idx]

    serialisable_ds_args = {
        **ds_args,
        "blur_params": ds_args["blur_params"]._asdict(),
    }

    run_config = lighter.Config(dict(CONFIG))
    run_config["dataset"] = {**CONFIG.dataset, **serialisable_ds_args}

    ds_root = root / f"ds_{ds_idx:02d}"
    ds_root.mkdir(parents=True, exist_ok=True)

    report_path = ds_root / CONFIG.report.format(time=time)

    print("=" * 80)
    print(f"DATASET {ds_idx}: {serialisable_ds_args}")
    print("=" * 80)

    dataset = Flickr2K(**run_config.dataset, root=Path("./data/").resolve())

    gkf = GroupKFold(**run_config.validation)
    train_idx, val_idx = next(iter(gkf.split(dataset.samples, groups=dataset.groups)))

    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True,  **run_config.dataloader)
    val_loader   = DataLoader(Subset(dataset, val_idx),   shuffle=False, **run_config.dataloader)

    model = UformerB(**run_config.model)
    model.compile(
        torch.optim.Adam(model.parameters(), **run_config.optimizer),
        torch.nn.MSELoss(),
        metrics=[PSNR()],
        device=device,
    )
    model.load(WEIGHTS_PATH)

    hist_params = dict(
        epochs=1,
        steps=len(train_loader),
        val_steps=len(val_loader),
        val_freq=1,
    )

    model.evaluate(
        data_loader=val_loader, callbacks=[Reporter(report_path, hist_params)]
    )


if __name__ == "__main__":
    freeze_support()
    app()