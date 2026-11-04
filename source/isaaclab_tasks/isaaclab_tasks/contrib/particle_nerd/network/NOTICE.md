# Architecture attribution

Slice attention follows the standard Transolver equations and the Torch path in:

- https://github.com/mnabian/physicsnemo/blob/fa1d68a67cc77baa176db8b4c1c5ef2ecdce00f7/physicsnemo/nn/module/physics_attention.py
- https://github.com/mnabian/physicsnemo/blob/fa1d68a67cc77baa176db8b4c1c5ef2ecdce00f7/physicsnemo/models/transolver/transolver.py
- Original Transolver: https://github.com/thuml/Transolver
- Crash application: https://arxiv.org/abs/2510.15201

Our particle/collider adapter, displacement integration, training code, and Warp
features are local adaptations. Checkpoints are not compatible with the reference
model. Domain parallelism, Transformer Engine, structured meshes, GeoTransolver,
and Transolver++ are not included.

The upstream Transolver source carries this license, retained for the adapted
attention structure:

MIT License

Copyright (c) 2024 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
