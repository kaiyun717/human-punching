"""Select NVIDIA EGL by the allocated CUDA UUID, never by Slurm ordinals.

Adapted from robust_sav's accepted boxing/perception/egl.py identity query.
Mesa and NVIDIA can advertise the same DRM device. Only EGL_NV_device_cuda
supplies the driver handle used here; a duplicate DRM advertisement is ignored.
No display is initialized on an unallocated device.
"""
from __future__ import annotations

import ctypes
import os
import uuid


def select_identity(records, allocated_uuid):
    matches = [row for row in records if row["gpu_uuid"] == allocated_uuid]
    if len(matches) != 1:
        raise RuntimeError("Expected one NVIDIA EGL device matching allocated CUDA UUID")
    return matches[0].copy()


def configure_allocated_egl():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("EGL requires an explicit Slurm GPU allocation")
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This renderer requires exactly one allocated CUDA GPU")
    identifier = torch.cuda.get_device_properties(0).uuid
    if isinstance(identifier, bytes):
        identifier = identifier.decode("ascii")
    allocated = "GPU-"+str(uuid.UUID(str(identifier).removeprefix("GPU-")))
    from OpenGL import EGL
    query_devices = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))(
        EGL.eglGetProcAddress("eglQueryDevicesEXT"))
    query_attribute = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ssize_t))(EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))
    count = ctypes.c_int()
    if not query_devices(0, None, ctypes.byref(count)) or not 0 < count.value <= 64:
        raise RuntimeError("Invalid EGL device enumeration")
    devices = (ctypes.c_void_p*count.value)()
    if not query_devices(count.value, devices, ctypes.byref(count)):
        raise RuntimeError("EGL device query failed")
    driver = ctypes.CDLL("libcuda.so.1")
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
    get_uuid.argtypes, get_uuid.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
    records = []
    for index, device in enumerate(devices):
        handle = ctypes.c_ssize_t()
        success = query_attribute(device, 0x323A, ctypes.byref(handle))
        EGL.eglGetError()
        if not success:
            continue
        raw = (ctypes.c_ubyte*16)()
        if get_uuid(ctypes.byref(raw), int(handle.value)) == 0:
            records.append({"egl_device_index": index, "cuda_handle": int(handle.value),
                "gpu_uuid": "GPU-"+str(uuid.UUID(bytes=bytes(raw)))})
    selected = select_identity(records, allocated)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(selected["egl_device_index"])
    from mujoco import egl
    from OpenGL import GL
    context = egl.GLContext(1, 1)
    try:
        context.make_current()
        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
        if "nvidia" not in vendor.lower() or any(x in renderer.lower() for x in ("llvmpipe", "softpipe", "software")):
            raise RuntimeError("GPU hardware rendering was not established")
    finally:
        context.free()
    return {**selected, "vendor": vendor, "renderer": renderer,
            "identity_method": "EGL_CUDA_DEVICE_NV handle to CUDA UUID",
            "enumerated_cuda_devices": records}
