"""Patch a run's latest checkpoint with overrides on non-architectural model fields.

The resume-time guard compares every field of RollingFlowConfig, so both the
checkpoint's embedded config and the YAML must be updated to agree. Fields that
shape weights are rejected up-front — patching them here would pass the guard
but then trip load_state_dict.

    uv run python scripts/patch_run.py SRC DST \\
        --set n_denoising_steps=32 --set noise_scale=0.9

Then resume:
    uv run python scripts/train.py --config_path DST/config.yaml
"""

import argparse
import dataclasses
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import yaml

# Fields whose value shapes or anchors the model weights. Changing any of these
# would corrupt a resumed run (load_state_dict would fail or the codec/loss
# would no longer match what the weights learned), so they're forbidden here.
_ARCHITECTURAL_FIELDS = frozenset(
    {"transformer_config", "vocabulary_size", "acoustic_dim", "codec", "parametrization"}
)


def _cast(field_type: Any, raw: str):
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return field_type[raw]
    if field_type is bool:
        return raw.lower() in ("1", "true", "yes", "y")
    return field_type(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Source run directory")
    parser.add_argument("dst", type=Path, help="Destination run directory (must not exist)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="override a top-level RollingFlowConfig field (repeatable)",
    )
    args = parser.parse_args()

    if not args.overrides:
        parser.error("at least one --set FIELD=VALUE is required")

    src_latest = args.src / "checkpoints" / "checkpoint.latest.pt"
    src_config = args.src / "config.yaml"
    if not src_latest.exists():
        raise FileNotFoundError(src_latest)
    if not src_config.exists():
        raise FileNotFoundError(src_config)
    if args.dst.exists():
        raise FileExistsError(f"{args.dst} already exists; refusing to clobber")

    overrides = dict(pair.split("=", 1) for pair in args.overrides)
    src_ckpt = src_latest.resolve()
    print(f"Source checkpoint: {src_ckpt.name}")

    data = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    if "config" not in data:
        raise RuntimeError("source checkpoint has no embedded model config")
    cfg_obj = data["config"]
    field_types = {f.name: f.type for f in dataclasses.fields(cfg_obj)}

    casted: dict[str, object] = {}
    for name, raw in overrides.items():
        if name not in field_types:
            raise KeyError(f"unknown RollingFlowConfig field: {name}")
        if name in _ARCHITECTURAL_FIELDS:
            raise ValueError(
                f"refusing to override architectural field {name!r}: changing it "
                "would break load_state_dict on resume"
            )
        casted[name] = _cast(field_types[name], raw)
        old = getattr(cfg_obj, name)
        setattr(cfg_obj, name, casted[name])
        print(f"  {name}: {old!r} -> {casted[name]!r}")

    dst_ckpt_dir = args.dst / "checkpoints"
    dst_ckpt_dir.mkdir(parents=True)
    dst_ckpt = dst_ckpt_dir / src_ckpt.name
    torch.save(data, dst_ckpt)
    (dst_ckpt_dir / "checkpoint.latest.pt").symlink_to(src_ckpt.name)

    cfg = yaml.safe_load(src_config.read_text())
    for name, value in casted.items():
        cfg["model"][name] = value.name if isinstance(value, Enum) else value
    cfg["output_dir"] = str(args.dst)
    cfg["resume"] = True
    (args.dst / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(
        f"Done. Resume with:\n"
        f"  uv run python scripts/train.py --config_path {args.dst}/config.yaml"
    )


if __name__ == "__main__":
    main()
