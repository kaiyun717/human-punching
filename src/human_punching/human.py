"""Plain light-blue mannequin surface for a MuJoCo motion-retargeting preview.

The mesh is procedural and redistributable with this project. It is not SMPL,
SMPL-X, a scan, or a physically calibrated human model. Each simplified surface
is a triangle mesh rigidly attached to one of the 21 DuoBox joints. This keeps
the exact source joint rotations inspectable without requiring body-model files.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .motion import JOINT_NAMES, PARENTS

# qpos = translation, root quaternion, 20 ball quaternions.
SOURCE_TO_WORLD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
SOURCE_TO_WORLD_QUAT = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])


def _fmt(values) -> str:
    return " ".join(f"{float(x):.10g}" for x in values)


def _profile(origin, axis, levels, widths, depths, *, centers=None, rings=32):
    """Closed smooth ring-profile surface; radii refer to two transverse axes."""
    origin = np.asarray(origin, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(axis @ candidate) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    e1 = candidate - axis * (axis @ candidate)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    angles = np.arange(rings) * (2 * np.pi / rings)
    vertices = []
    for i, (level, width, depth) in enumerate(zip(levels, widths, depths, strict=True)):
        center = origin + float(level) * axis
        if centers is not None:
            center += np.asarray(centers[i], dtype=float)
        vertices.extend(
            center + np.outer(np.cos(angles) * width, e1) + np.outer(np.sin(angles) * depth, e2)
        )
    faces = []
    for i in range(len(levels) - 1):
        for j in range(rings):
            a, b = i * rings + j, i * rings + (j + 1) % rings
            c, d = (i + 1) * rings + (j + 1) % rings, (i + 1) * rings + j
            faces.extend(((a, b, c), (a, c, d)))
    for level_index, lower in ((0, True), (len(levels) - 1, False)):
        center_index = len(vertices)
        vertices.append(np.mean(vertices[level_index * rings : (level_index + 1) * rings], axis=0))
        for j in range(rings):
            a, b = level_index * rings + j, level_index * rings + (j + 1) % rings
            faces.append((center_index, b, a) if lower else (center_index, a, b))
    return np.asarray(vertices), np.asarray(faces, dtype=int)


def _ellipsoid(center, radii, *, rings=32, rows=18):
    lat = np.linspace(-np.pi / 2 + 0.0001, np.pi / 2 - 0.0001, rows)
    rx, ry, rz = radii
    return _profile(
        center, [0.0, 1.0, 0.0], ry * np.sin(lat), rx * np.cos(lat), rz * np.cos(lat), rings=rings
    )


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray):
    normals = np.zeros_like(vertices)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norm, 1e-12)
    with path.open("w") as out:
        out.write("# Human Punching: authored procedural human surface; meters\n")
        for v in vertices:
            out.write("v " + _fmt(v) + "\n")
        for n in normals:
            out.write("vn " + _fmt(n) + "\n")
        for face in faces + 1:
            out.write("f " + " ".join(f"{i}//{i}" for i in face) + "\n")


def build_human(output_dir: Path, rest_offsets: np.ndarray, height_m: float = 1.8) -> dict:
    """Write an articulated human MJCF plus OBJ meshes and return its metadata.

    ``rest_offsets`` are the 21 joint-local bind offsets in the DuoBox order,
    with Y up, X anatomically left and Z forward. The root offset is ignored.
    The complete neutral mesh, including sole and crown, is scaled to height_m.
    ``target_offsets`` and ``scale`` must also be used when retargeting motion.
    The model's neutral root has a +90 degree X rotation to MuJoCo's Z-up frame.
    """
    output_dir = Path(output_dir).resolve()
    offsets = np.asarray(rest_offsets, dtype=float).copy()
    if offsets.shape != (21, 3) or not np.isfinite(offsets).all():
        raise ValueError("rest_offsets must be a finite (21, 3) array")
    if not np.isfinite(height_m) or height_m <= 0:
        raise ValueError("height_m must be positive and finite")
    offsets[0] = 0.0
    leg_length = np.linalg.norm(offsets[14]) + np.linalg.norm(offsets[15])
    if leg_length <= 1e-8:
        raise ValueError("The source skeleton must have nonzero leg lengths")
    # Morphological dimensions use the source's leg length, then every dimension
    # undergoes the same final crown-to-sole scale (no independent limb scaling).
    u = leg_length / 0.86
    meshes = []

    def surface(body, name, material, data):
        vertices, faces = data
        meshes.append(dict(body=body, name=name, material=material, vertices=vertices, faces=faces))

    def ell(body, name, material, center, radii):
        surface(body, name, material, _ellipsoid(np.asarray(center) * u, np.asarray(radii) * u))

    def profile(
        body,
        name,
        material,
        levels,
        widths,
        depths,
        *,
        origin=(0, 0, 0),
        axis=(0, 1, 0),
        centers=None,
    ):
        surface(
            body,
            name,
            material,
            _profile(
                np.asarray(origin) * u,
                axis,
                np.asarray(levels) * u,
                np.asarray(widths) * u,
                np.asarray(depths) * u,
                centers=None if centers is None else np.asarray(centers) * u,
            ),
        )

    # A featureless mannequin: one body color, smooth head, simple solid feet.
    # Keep source joint frames even where there is no separately visible part.
    spine_len = np.linalg.norm(offsets[2]) / u
    chest_len = np.linalg.norm(offsets[3]) / u
    profile(
        "Hips",
        "pelvis",
        "body",
        [-0.13, -0.09, -0.01, 0.055, 0.085],
        [0.10, 0.148, 0.168, 0.155, 0.142],
        [0.075, 0.10, 0.108, 0.095, 0.088],
    )
    profile(
        "Spine",
        "abdomen",
        "body",
        [-0.035, 0, spine_len * 0.5, spine_len, spine_len + 0.02],
        [0.135, 0.14, 0.148, 0.165, 0.17],
        [0.087, 0.09, 0.096, 0.102, 0.10],
    )
    profile(
        "Spine1",
        "torso",
        "body",
        [-0.04, 0, chest_len * 0.45, chest_len * 0.78, chest_len + 0.015],
        [0.159, 0.173, 0.183, 0.159, 0.063],
        [0.095, 0.103, 0.108, 0.09, 0.055],
    )
    neck_len = np.linalg.norm(offsets[4]) / u
    profile(
        "Neck",
        "neck_surface",
        "body",
        [-0.025, 0.01, neck_len * 0.55, neck_len * 0.8],
        [0.058, 0.05, 0.044, 0.041],
        [0.05, 0.047, 0.043, 0.04],
    )
    ell("Head", "head_surface", "body", [0, 0.018, 0], [0.079, 0.127, 0.083])

    for sign, side, shoulder, upper, fore, hand, thigh, shin, foot, toe in (
        (1, "Left", 5, 6, 7, 8, 13, 14, 15, 16),
        (-1, "Right", 9, 10, 11, 12, 17, 18, 19, 20),
    ):
        shoulder_len = np.linalg.norm(offsets[upper]) / u
        profile(
            JOINT_NAMES[shoulder],
            f"{side}_shoulder",
            "body",
            [-0.012, shoulder_len * 0.45, shoulder_len + 0.016],
            [0.042, 0.052, 0.061],
            [0.047, 0.054, 0.058],
            axis=(sign, 0, 0),
        )
        ell(JOINT_NAMES[upper], f"{side}_arm_joint", "body", [0, 0, 0], [0.063, 0.06, 0.06])
        length = np.linalg.norm(offsets[fore]) / u
        profile(
            JOINT_NAMES[upper],
            f"{side}_upper_arm",
            "body",
            [-0.025, 0.055, length * 0.6, length + 0.012],
            [0.052, 0.056, 0.051, 0.038],
            [0.05, 0.052, 0.049, 0.039],
            axis=(sign, 0, 0),
        )
        ell(JOINT_NAMES[fore], f"{side}_elbow", "body", [0, 0, 0], [0.04, 0.04, 0.04])
        length = np.linalg.norm(offsets[hand]) / u
        profile(
            JOINT_NAMES[fore],
            f"{side}_forearm",
            "body",
            [-0.014, 0.05, length * 0.55, length + 0.014],
            [0.037, 0.043, 0.037, 0.026],
            [0.038, 0.043, 0.037, 0.027],
            axis=(sign, 0, 0),
        )
        glove_material = "left_glove" if sign == 1 else "right_glove"
        profile(
            JOINT_NAMES[hand],
            f"{side}_glove_cuff",
            "glove_cuff",
            [-0.047, -0.027, 0.025],
            [0.037, 0.042, 0.041],
            [0.039, 0.044, 0.043],
            axis=(sign, 0, 0),
        )
        # A flattened rounded block and pale back-of-hand patch expose rotation.
        # Both are rigidly attached to the complete source Hand orientation.
        vertices, faces = _ellipsoid([0, 0, 0], [1, 1, 1])
        vertices = np.sign(vertices) * np.abs(vertices) ** 0.45
        vertices = (vertices * [0.078, 0.050, 0.061] + [sign * 0.062, -0.004, 0.01]) * u
        surface(JOINT_NAMES[hand], f"{side}_boxing_glove", glove_material, (vertices, faces))
        ell(
            JOINT_NAMES[hand],
            f"{side}_glove_mark",
            "glove_mark",
            [sign * 0.063, 0.047, 0.01],
            [0.041, 0.006, 0.024],
        )
        length = np.linalg.norm(offsets[shin]) / u
        profile(
            JOINT_NAMES[thigh],
            f"{side}_thigh",
            "body",
            [-0.025, 0.065, length * 0.65, length + 0.015],
            [0.078, 0.085, 0.07, 0.048],
            [0.078, 0.088, 0.075, 0.05],
            axis=(0, -1, 0),
        )
        ell(JOINT_NAMES[shin], f"{side}_knee", "body", [0, 0, 0], [0.05, 0.05, 0.05])
        length = np.linalg.norm(offsets[foot]) / u
        profile(
            JOINT_NAMES[shin],
            f"{side}_lower_leg",
            "body",
            [-0.016, 0.08, length * 0.55, length + 0.012],
            [0.044, 0.056, 0.045, 0.03],
            [0.045, 0.057, 0.046, 0.031],
            axis=(0, -1, 0),
        )
        toe_offset = offsets[toe] / u
        ell(
            JOINT_NAMES[foot],
            f"{side}_foot",
            "body",
            [0, toe_offset[1] * 0.65, toe_offset[2] * 0.48],
            [0.05, 0.045, 0.125],
        )

    bind_positions = np.zeros((21, 3))
    for j in range(1, 21):
        bind_positions[j] = bind_positions[PARENTS[j]] + offsets[j]
    all_vertices = np.concatenate(
        [m["vertices"] + bind_positions[JOINT_NAMES.index(m["body"])] for m in meshes]
    )
    source_min_y = float(all_vertices[:, 1].min())
    source_max_y = float(all_vertices[:, 1].max())
    scale = float(height_m / (source_max_y - source_min_y))
    root_height = -source_min_y * scale
    offsets *= scale
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    for mesh in meshes:
        _write_obj(mesh_dir / f"{mesh['name']}.obj", mesh["vertices"] * scale, mesh["faces"])

    root = ET.Element("mujoco", model="HumanPunching_procedural_surface")
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        meshdir="meshes",
        autolimits="true",
        inertiafromgeom="false",
    )
    ET.SubElement(root, "option", timestep="0.008333333333", gravity="0 0 0", integrator="Euler")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual, "global", offwidth="1920", offheight="1080", azimuth="135", elevation="-10"
    )
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(
        visual,
        "headlight",
        ambient="0.35 0.35 0.35",
        diffuse="0.65 0.65 0.65",
        specular="0.15 0.15 0.15",
    )
    ET.SubElement(visual, "map", znear="0.01", zfar="40", fogstart="8", fogend="20")
    ET.SubElement(visual, "rgba", haze="0.12 0.15 0.20 1")
    ET.SubElement(root, "statistic", center="0 0 0.9", extent="2.0")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(
        assets,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.12 0.16 0.21",
        rgb2="0.03 0.04 0.07",
        width="512",
        height="3072",
    )
    materials = {
        "body": ("0.39 0.70 0.91 1", ".10", ".18"),
        "left_glove": ("0.055 0.28 0.62 1", ".18", ".28"),
        "right_glove": ("0.70 0.12 0.10 1", ".18", ".28"),
        "glove_cuff": ("0.055 0.08 0.12 1", ".08", ".16"),
        "glove_mark": ("0.84 0.94 1.0 1", ".08", ".16"),
    }
    for name, (rgba, specular, shininess) in materials.items():
        ET.SubElement(
            assets, "material", name=name, rgba=rgba, specular=specular, shininess=shininess
        )
    ET.SubElement(
        assets,
        "texture",
        name="floor_tex",
        type="2d",
        builtin="checker",
        rgb1="0.14 0.18 0.23",
        rgb2="0.17 0.21 0.27",
        width="512",
        height="512",
    )
    ET.SubElement(
        assets,
        "material",
        name="floor_mat",
        texture="floor_tex",
        texrepeat="2 2",
        reflectance="0.06",
        texuniform="true",
    )
    for mesh in meshes:
        ET.SubElement(assets, "mesh", name=mesh["name"], file=f"{mesh['name']}.obj")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "geom", contype="0", conaffinity="0", group="0")
    ET.SubElement(defaults, "joint", damping="0", stiffness="0", armature="0")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        name="key",
        pos="-3 -4 5",
        dir="0.35 0.45 -0.8",
        directional="true",
        diffuse="0.85 0.83 0.80",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "light",
        name="fill",
        pos="3 1 4",
        dir="-0.4 -0.1 -0.9",
        directional="true",
        diffuse="0.42 0.5 0.65",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="6 6 .05",
        material="floor_mat",
        pos="0 0 -0.003",
    )
    bodies = []
    masses = [
        10,
        8,
        14,
        1,
        4.5,
        0.3,
        2.2,
        1.3,
        0.8,
        0.3,
        2.2,
        1.3,
        0.8,
        7.5,
        3.8,
        0.8,
        0.2,
        7.5,
        3.8,
        0.8,
        0.2,
    ]
    for j, name in enumerate(JOINT_NAMES):
        parent = world if j == 0 else bodies[PARENTS[j]]
        body = ET.SubElement(
            parent, "body", name=name, pos=_fmt([0, 0, root_height] if j == 0 else offsets[j])
        )
        if j == 0:
            body.set("quat", _fmt(SOURCE_TO_WORLD_QUAT))
            ET.SubElement(body, "freejoint", name="root")
        else:
            ET.SubElement(body, "joint", name=f"{name}_joint", type="ball")
        # Nonzero explicit inertias make the MJCF structurally valid; these are
        # placeholders, not suitable for training a dynamics-controlled policy.
        inertia = max(masses[j] * 0.001, 0.0001)
        ET.SubElement(
            body, "inertial", pos="0 0 0", mass=str(masses[j]), diaginertia=_fmt([inertia] * 3)
        )
        for mesh in meshes:
            if mesh["body"] == name:
                ET.SubElement(
                    body,
                    "geom",
                    name=mesh["name"],
                    type="mesh",
                    mesh=mesh["name"],
                    material=mesh["material"],
                )
        bodies.append(body)
    xml_path = output_dir / "human.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="unicode", xml_declaration=True)
    metadata = {
        "xml_path": str(xml_path),
        "height_m": float(height_m),
        "standing_height_m": (source_max_y - source_min_y) * scale,
        "scale": scale,
        "source_height_m": source_max_y - source_min_y,
        "root_height_m": root_height,
        "neutral_min_y": source_min_y * scale,
        "neutral_max_y": source_max_y * scale,
        "target_offsets": offsets.tolist(),
        "joint_names": list(JOINT_NAMES),
        "parents": PARENTS.tolist(),
        "source_to_world": SOURCE_TO_WORLD.tolist(),
        "source_to_world_quat_wxyz": SOURCE_TO_WORLD_QUAT.tolist(),
        "appearance": "Plain light-blue mannequin, featureless head, no clothing or separate toes.",
        "glove_attachment": "All glove geoms belong to the corresponding Hand body; asymmetric patch shows orientation.",
        "mesh_count": len(meshes),
        "vertex_count": sum(len(m["vertices"]) for m in meshes),
        "triangle_count": sum(len(m["faces"]) for m in meshes),
        "qpos_size": 87,
        "provenance": "Authored light-blue featureless mannequin triangle meshes; no SMPL/SMPL-X assets or parameters.",
        "limitations": [
            "Segmented rigid surfaces, not a skinned deformable body.",
            "Placeholder inertial parameters; kinematic replay only.",
            "Gravity and all contact masks are disabled; no contact dynamics or learned controller.",
        ],
        "height_definition": "World-Z maximum minus minimum of all neutral triangle-mesh vertices, crown to sole.",
    }
    (output_dir / "human_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
