# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Freeze the approved local backend and source into a portable cloud bundle."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isaaclab", type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bundle = args.output / "bundle"
    bundle.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".venv", "data", "logs", "outputs")
    shutil.copytree(args.isaaclab / "source", bundle / "isaaclab/source", ignore=ignore)
    # Core asset-root discovery reads apps/isaaclab.python.kit even for
    # kitless Newton launches. Keep the complete small launcher config set.
    shutil.copytree(args.isaaclab / "apps", bundle / "isaaclab/apps", ignore=ignore)
    shutil.copytree(args.isaaclab / "examples/particle_nerd", bundle / "isaaclab/examples/particle_nerd", ignore=ignore)
    for filename in ("LICENSE", "VERSION"):
        if (args.isaaclab / filename).is_file():
            shutil.copy2(args.isaaclab / filename, bundle / "isaaclab" / filename)
    vendor = bundle / "vendor"
    packages = {}
    # Preserve the exact tested fork and Warp native libraries. No rebuilding,
    # resolving another branch head, or changing solver implementations.
    for name in ("newton", "warp-lang", "newton-usd-schemas"):
        dist = metadata.distribution(name)
        packages[name] = dict(version=dist.version, direct_url=json.loads(dist.read_text("direct_url.json") or "null"))
        for relative in dist.files or ():
            if ".." in relative.parts or "__pycache__" in relative.parts or relative.suffix == ".pyc":
                continue
            source = Path(dist.locate_file(relative))
            if source.is_file():
                target = vendor / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    names = [
        "numpy",
        "torch",
        "mujoco",
        "mujoco-warp",
        "usd-core",
        "h5py",
        "gymnasium",
        "prettytable",
        "psutil",
        "trimesh",
        "scipy",
        "pyglet",
        "PyYAML",
        "hydra-core",
        "rich",
        "tqdm",
        "gitpython",
        "packaging",
        "pillow",
        "matplotlib",
        "cbor2",
        "lazy-loader",
        "leapp",
        "onnx",
        "onnxruntime",
        "typing-extensions",
        "einops",
        "omniverseclient",
        "requests",
    ]
    versions = {name: metadata.version(name) for name in names}
    (bundle / "requirements.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n"
    )
    files = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                files[path.relative_to(bundle).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        isaaclab_source=str(args.isaaclab),
        isaaclab_commit=subprocess.check_output(
            ["git", "-C", str(args.isaaclab), "rev-parse", "HEAD"], text=True
        ).strip(),
        vendor_packages=packages,
        requirements=versions,
        files_sha256=files,
        asset_note=(
            "Primitive box and procedural material; GroundPlaneCfg downloads IsaacLab's "
            "default checker-plane USD through omniverseclient. No robot assets "
            "required."
        ),
    )
    (bundle / "snapshot.json").write_text(json.dumps(provenance, indent=2) + "\n")
    archive = args.output / "bundle.tar.gz"
    # Corporate NSS name lookup may block for local UID/GID values; numeric
    # ownership is sufficient because the cloud task extracts its own files.
    tarfile.pwd = None
    tarfile.grp = None
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        tar.add(bundle, arcname="bundle")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (args.output / "bundle.sha256").write_text(f"{digest}  bundle.tar.gz\n")
    print(json.dumps(dict(archive=str(archive), sha256=digest, bytes=archive.stat().st_size)))


if __name__ == "__main__":
    main()
