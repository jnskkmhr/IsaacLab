# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the exact GPU visibility used by region campaign subprocesses."""

import json
import os
import subprocess
import sys


def main():
    probe = """import json, torch
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
x = torch.ones(1, device="cuda:0")
torch.cuda.synchronize()
p = torch.cuda.get_device_properties(0)
print(json.dumps(dict(name=p.name, uuid=str(getattr(p, "uuid", "")), value=x.item())))
"""
    devices = []
    for gpu in range(4):
        result = subprocess.run(
            [sys.executable, "-c", probe],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
            check=True,
            capture_output=True,
            text=True,
        )
        devices.append(dict(index=gpu, **json.loads(result.stdout.strip())))
    uuids = [d["uuid"] for d in devices if d["uuid"]]
    if uuids and len(set(uuids)) != 4:
        raise RuntimeError(f"GPU workers do not map to four distinct devices: {devices}")
    print(json.dumps(dict(gpu_visibility_verified=devices)), flush=True)


if __name__ == "__main__":
    main()
