"""Export an inspectable D1 MJCF with the mounted emitters and the stock flat floor."""
from argparse import ArgumentParser
from pathlib import Path
from xml.etree import ElementTree as ET

from tools94.common94 import ROOT, digest, read, write
from tools107.capture107 import geometry
from tools107.mount107 import CONFIG, _mounted_scene, parts
from tools107.run107 import verify_vendor_geometry


def export(destination, arm):
    cfg = read(ROOT / "training/configs/pilot03d.json")
    plane = ROOT / cfg["scene"]
    verify_vendor_geometry()
    if digest(plane) != cfg["scene_sha256"]:
        raise ValueError("the registered flat scene has changed")
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=False)
    try:
        model_dir = destination / "mujoco"
        model_dir.mkdir()
        (destination / "meshes").symlink_to(plane.parent.parent / "meshes", target_is_directory=True)
        (model_dir / "robot.xml").symlink_to(plane.with_name("robot.xml"))
        source = model_dir / "source_plane.xml"
        source.symlink_to(plane)
        scene = _mounted_scene(source, geometry(), "flat", arm)
        robot = scene.with_name("robot107.xml")
        base = ET.parse(robot).find("./worldbody/body[@name='base_link']")
        if {s.get("name") for s in base.findall("site") if s.get("name", "").startswith("mount107_")} != set(read(CONFIG)["site_names"]):
            raise ValueError("missing optical site in the exported model")
        if {g.get("name") for g in base.findall("geom") if g.get("name", "").startswith("mount107_")} != set(parts(arm)):
            raise ValueError("incomplete support in the exported model")
        write(destination / "manifest.json", dict(arm=arm, scene_sha256=digest(scene), robot_sha256=digest(robot),
              mount_definition_sha256=digest(CONFIG), vendor_robot_sha256=cfg["robot_xml_sha256"], physics_steps=0, rays=0))
        return scene
    except BaseException as exc:
        write(destination / "partial.json", {"status":"PARTIAL", "error":repr(exc)})
        raise


def check_local(destination):
    import mujoco
    import numpy as np
    from tools107.mount107 import mount_positions
    root = Path(destination)
    receipt = read(root / "manifest.json")
    scene = root / "mujoco/scene107.xml"
    if digest(scene) != receipt["scene_sha256"] or digest(scene.with_name("robot107.xml")) != receipt["robot_sha256"]:
        raise ValueError("the exported model differs from its receipt")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    base = model.body("base_link").id
    sites = [model.site(name).id for name in read(CONFIG)["site_names"]]
    desired = mount_positions(receipt["arm"]) @ data.xmat[base].reshape(3, 3).T + data.xpos[base]
    np.testing.assert_allclose(data.site_xpos[sites], desired, rtol=0., atol=1e-10)
    if data.time != 0. or not np.all(model.site_bodyid[sites] == base):
        raise ValueError("emitter frame or simulation clock changed")
    return f"{scene}: {len(sites)} fixed sites, MuJoCo {mujoco.__version__}, zero simulation time"


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("destination", type=Path, help="new local folder; original vendor meshes are linked in place")
    p.add_argument("--arm", choices=("original", "forward"), default="forward")
    p.add_argument("--check", action="store_true", help="load and verify an already exported model")
    args = p.parse_args()
    print(check_local(args.destination) if args.check else export(args.destination, args.arm))
