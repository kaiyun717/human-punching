#!/usr/bin/env python3
"""Verify native MuJoCo EGL rendering on a GPU visible to this Slurm job.

CUDA logical ordinals and EGL device indices are different namespaces. Match
physical PCI addresses before selecting an EGL device; never guess index zero.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


class PreflightError(RuntimeError):
    """The allocation or renderer could not be verified."""


def normalize_pci(value: str) -> str:
    match = re.fullmatch(r"([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value)
    if not match:
        raise PreflightError(f"Invalid PCI address: {value!r}")
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function}"


def cuda_devices():
    """Use the driver API, which honors Slurm's CUDA_VISIBLE_DEVICES."""
    cuda = ctypes.CDLL("libcuda.so.1")

    def call(name, *args):
        status = getattr(cuda, name)(*args)
        if status:
            raise PreflightError(f"CUDA {name} failed with error {status}")

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    if count.value < 1:
        raise PreflightError("CUDA sees no GPU in this Slurm allocation")

    def pci_for_handle(handle):
        buffer = ctypes.create_string_buffer(64)
        call("cuDeviceGetPCIBusId", buffer, len(buffer), ctypes.c_int(handle))
        return normalize_pci(buffer.value.decode())

    records = []
    for ordinal in range(count.value):
        handle = ctypes.c_int()
        call("cuDeviceGet", ctypes.byref(handle), ordinal)
        name = ctypes.create_string_buffer(256)
        call("cuDeviceGetName", name, len(name), handle)
        major, minor = ctypes.c_int(), ctypes.c_int()
        call("cuDeviceComputeCapability", ctypes.byref(major), ctypes.byref(minor), handle)
        identifier = (ctypes.c_ubyte * 16)()
        call("cuDeviceGetUuid", ctypes.byref(identifier), handle)
        records.append(
            {
                "logical_ordinal": ordinal,
                "handle": handle.value,
                "name": name.value.decode(),
                "pci_bus_id": pci_for_handle(handle.value),
                "uuid": f"GPU-{uuid.UUID(bytes=bytes(identifier))}",
                "compute_capability": f"{major.value}.{minor.value}",
            }
        )
    return records, pci_for_handle


def drm_pci(device_file: str | None) -> str | None:
    if not device_file:
        return None
    device_path = Path("/sys/class/drm") / Path(device_file).name / "device"
    if not device_path.exists():
        return None
    return normalize_pci(device_path.resolve().name)


def egl_function(egl, name, result_type, *argument_types):
    address = egl.eglGetProcAddress(name.encode())
    if not address:
        raise PreflightError(f"Missing required EGL entry point {name}")
    return ctypes.CFUNCTYPE(result_type, *argument_types)(address)


def choose_egl_device(egl, target_pci, pci_for_handle):
    query_string = egl_function(
        egl, "eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int
    )
    query_attribute = egl_function(
        egl,
        "eglQueryDeviceAttribEXT",
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ssize_t),
    )
    devices = egl.eglQueryDevicesEXT()
    records = []
    matches = []
    for index, device in enumerate(devices):
        extension_bytes = query_string(device, 0x3055)  # EGL_EXTENSIONS
        extensions = extension_bytes.decode().split() if extension_bytes else []
        record = {"index": index, "extensions": extensions}
        addresses = set()
        for extension, token, key in (
            ("EGL_EXT_device_drm_render_node", 0x3377, "drm_render_node"),
            ("EGL_EXT_device_drm", 0x3233, "drm_device"),
        ):
            if extension in extensions:
                value = query_string(device, token)
                record[key] = value.decode() if value else None
                pci = drm_pci(record[key])
                if pci:
                    addresses.add(pci)
        if addresses:
            record["mapping_method"] = "DRM sysfs PCI address"
        elif "EGL_NV_device_cuda" in extensions:
            # The extension specifies a CUDA API-compatible device handle.
            # Resolve it through CUDA, never assume it is a physical GPU index.
            handle = ctypes.c_ssize_t()
            if query_attribute(device, 0x323A, ctypes.byref(handle)):
                record["cuda_handle"] = handle.value
                try:
                    addresses.add(pci_for_handle(handle.value))
                    record["mapping_method"] = "EGL_NV_device_cuda handle to CUDA PCI address"
                except PreflightError as exc:
                    record["mapping_error"] = str(exc)
            else:
                record["mapping_error"] = "EGL_CUDA_DEVICE_NV query failed"
                egl.eglGetError()  # Clear an expected inaccessible-device error.
        record["pci_bus_ids"] = sorted(addresses)
        records.append(record)
        if addresses == {target_pci}:
            matches.append(index)
    if len(matches) != 1:
        raise PreflightError(
            f"Expected exactly one EGL device matching allocated CUDA PCI {target_pci}; "
            f"found {matches}. Enumerated devices: {json.dumps(records)}"
        )
    index = matches[0]
    return index, devices[index], records


def run(report):
    if not os.environ.get("SLURM_JOB_ID"):
        raise PreflightError("Run through sbatch/srun inside a Slurm GPU allocation")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or visible in {"-1", "NoDevFiles"}:
        raise PreflightError("Slurm must set a nonempty CUDA_VISIBLE_DEVICES; do not override it")
    # Set the rendering platform before any OpenGL or MuJoCo import.
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    devices, pci_for_handle = cuda_devices()
    report["cuda_devices"] = devices
    target = devices[0]
    report["target_cuda_device"] = target

    import mujoco
    import mujoco.egl
    import numpy as np
    from mujoco.egl import egl_ext as egl
    from OpenGL import GL

    report["mujoco_version"] = mujoco.__version__
    index, selected_device, egl_devices = choose_egl_device(
        egl, target["pci_bus_id"], pci_for_handle
    )
    report["egl_devices"] = egl_devices
    report["mujoco_egl_device_id"] = index
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(index)
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><visual><global offwidth="160" offheight="120"/></visual>
      <worldbody><light pos="0 -2 3"/><geom type="plane" size="2 2 .1"
      rgba=".3 .3 .3 1"/><geom type="sphere" pos="0 0 .3" size=".3"
      rgba=".2 .7 1 1"/></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=120, width=160) as renderer:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0, 0, 0.25]
        camera.distance = 2
        camera.azimuth = 135
        camera.elevation = -20
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        gl = {
            key: GL.glGetString(value).decode()
            for key, value in (
                ("vendor", GL.GL_VENDOR),
                ("renderer", GL.GL_RENDERER),
                ("version", GL.GL_VERSION),
            )
        }
        report["opengl"] = gl
        if "nvidia" not in gl["vendor"].lower():
            raise PreflightError(f"Expected NVIDIA hardware rendering; got {gl}")
        if any(word in gl["renderer"].lower() for word in ("llvmpipe", "softpipe", "software")):
            raise PreflightError(f"Software renderer is forbidden: {gl['renderer']}")
        query_display = egl_function(
            egl,
            "eglQueryDisplayAttribEXT",
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ssize_t),
        )
        actual_device = ctypes.c_ssize_t()
        if not query_display(mujoco.egl.EGL_DISPLAY, 0x322C, ctypes.byref(actual_device)):
            raise PreflightError("Could not verify the EGL display's physical device")
        selected_pointer = ctypes.cast(selected_device, ctypes.c_void_p).value
        if actual_device.value != selected_pointer:
            raise PreflightError("The active EGL display does not use the selected allocated GPU")
        report["egl_display_matches_selected_device"] = True
        report["frame"] = {
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "minimum": int(frame.min()),
            "maximum": int(frame.max()),
            "standard_deviation": float(frame.std()),
            "finite": bool(np.isfinite(frame).all()),
        }
        if not report["frame"]["finite"] or frame.std() < 1:
            raise PreflightError("MuJoCo rendered an invalid or constant test frame")
    report["status"] = "passed"
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON evidence, including failures"
    )
    parser.add_argument("--env-output", type=Path, help="Shell exports written only after success")
    args = parser.parse_args()
    report = {
        "status": "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_GPUS",
                "SLURM_STEP_GPUS",
                "SLURM_JOB_PARTITION",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "MUJOCO_GL",
                "MUJOCO_EGL_DEVICE_ID",
            )
        },
    }
    if args.env_output:
        args.env_output.unlink(missing_ok=True)
    try:
        index = run(report)
        if args.env_output:
            args.env_output.parent.mkdir(parents=True, exist_ok=True)
            args.env_output.write_text(
                f"export MUJOCO_GL=egl\nexport PYOPENGL_PLATFORM=egl\n"
                f"export MUJOCO_EGL_DEVICE_ID={index}\n"
            )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
