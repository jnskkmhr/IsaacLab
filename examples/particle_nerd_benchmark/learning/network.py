# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load the shared network sources without registering unrelated Gym tasks.

The current model is ``contrib/particle_nerd/network/nerd_network.py``; the superseded ones
remain in ``contrib/legacy/particle_nerd/network`` and
``contrib/legacy/particle_world/network``. Private package namespaces preserve their
relative imports while avoiding the eager environment discovery in
``isaaclab_tasks.__init__``. No implementation is copied.
"""

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

_PACKAGE = "_object_particle_nerd"
_SOURCE = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/legacy/particle_nerd"
if _PACKAGE not in sys.modules:
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_SOURCE)]
    package.__spec__ = importlib.machinery.ModuleSpec(_PACKAGE, loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[_PACKAGE] = package

_WORLD_PACKAGE = "_object_particle_world"
_WORLD_SOURCE = (
    Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/legacy/particle_world"
)
if _WORLD_PACKAGE not in sys.modules:
    world_pkg = types.ModuleType(_WORLD_PACKAGE)
    world_pkg.__path__ = [str(_WORLD_SOURCE)]
    world_pkg.__spec__ = importlib.machinery.ModuleSpec(_WORLD_PACKAGE, loader=None, is_package=True)
    world_pkg.__spec__.submodule_search_locations = world_pkg.__path__
    sys.modules[_WORLD_PACKAGE] = world_pkg

_NERD_PACKAGE = "_particle_nerd"
_NERD_SOURCE = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/particle_nerd"
if _NERD_PACKAGE not in sys.modules:
    nerd_pkg = types.ModuleType(_NERD_PACKAGE)
    nerd_pkg.__path__ = [str(_NERD_SOURCE)]
    nerd_pkg.__spec__ = importlib.machinery.ModuleSpec(_NERD_PACKAGE, loader=None, is_package=True)
    nerd_pkg.__spec__.submodule_search_locations = nerd_pkg.__path__
    sys.modules[_NERD_PACKAGE] = nerd_pkg

_features = importlib.import_module(_PACKAGE + ".network.features")
features = _features.features
material_features = _features.material_features
ParticleTransolver = importlib.import_module(_PACKAGE + ".network.model").ParticleTransolver

_world_cconv = importlib.import_module(_WORLD_PACKAGE + ".network.cconv_warp")
features_world_particle = _world_cconv.features_world_particle
WorldParticleNetwork = importlib.import_module(_WORLD_PACKAGE + ".network.model").WorldParticleNetwork

_nerd_network = importlib.import_module(_NERD_PACKAGE + ".network.nerd_network")
ParticleNeRDNetwork = _nerd_network.ParticleNeRDNetwork
PARTICLE_NERD_ARCHITECTURE = _nerd_network.ARCHITECTURE
PARTICLE_NERD_SOURCES = (
    _nerd_network,
    *(
        importlib.import_module(_NERD_PACKAGE + name)
        for name in (
            ".network.feature_builder",
            ".modules.cnn",
            ".modules.attention",
            ".modules.positional_encoder",
            ".modules.normalization",
            ".nerd_kernels",
        )
    ),
)
"""Modules that implement the Particle NeRD update; results are bound to the hash of their source files."""

_object_inference = importlib.import_module(_PACKAGE + ".network.object_inference")
ObjectParticleStepper = _object_inference.ObjectParticleStepper
OBJECT_PIPELINE = _object_inference.PIPELINE
OBJECT_SCHEMA = _object_inference.SCHEMA
CONTACT_SETTINGS = _object_inference.CONTACT_SETTINGS
validate_object_checkpoint = _object_inference.validate_checkpoint
ProxyEstimator = importlib.import_module(_PACKAGE + ".contact.proxy").ProxyEstimator


def nerd_source_hashes():
    """Bind results to every source file that implements the neural update."""
    import hashlib

    return {
        str(p.relative_to(_SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((_SOURCE / "network").glob("*.py"))
    }
