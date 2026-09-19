# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_neural' python package."""

import os
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Custom build command that bundles config/extension.toml into the package.

    This ensures the toml is available when installed as a regular (non-editable)
    wheel, e.g. when pulled in as a dependency via a file:// URL.
    """

    def run(self):
        super().run()
        src = os.path.join(EXTENSION_PATH, "config", "extension.toml")
        dst_dir = os.path.join(self.build_lib, "isaaclab_neural", "config")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, "extension.toml"))


EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

setup(cmdclass={"build_py": build_py})
