# Vendored ProtoMotions SOMA23 source

Source: NVIDIA ProtoMotions, revision `7a8417a9f55b1586b0d6e8f46578c5bc9d7ce1f1`.

Copyright (c) 2025–2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

The original `soma23_humanoid.xml` and `soma23_config.py` are retained unchanged, with their SHA-256 hashes and immutable URLs in `manifest.json`. The upstream Apache-2.0 license is retained as `LICENSE.md`; upstream asset-specific notices are retained as `THIRD_PARTY_NOTICES.txt`. No SMPL body-model data, pretrained policy, or photorealistic SOMA surface is included.

`src/human_punching/scalar_body.py` creates a **modified** derivative: reduced scalar articulation, changed axes/limits, primitive geometry, chosen segment masses, zero passive stiffness/damping, bounded gear-one torque motors, boxing gloves, floor/selected self-collision pairs, sites, camera and metadata. The generated body is called **SOMA23-constrained-48** and is not the unmodified upstream SOMA23 dynamics model. Original bone lengths and the 23-body hierarchy are retained.
