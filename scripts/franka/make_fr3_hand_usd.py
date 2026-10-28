"""Override layer for the FR3 + adapter + Sharpa robots of ~/workspace/fr3-sharpa-rubik: the hand joints get the official
float-base hand USD's drive / armature / friction (the URDF conversion left them 0), so the hand behaves as the one
the solve policies were trained with (masses and colliders already match: same official URDF meshes).

    python scripts/franka/make_fr3_hand_usd.py   ->  assets_franka/<side>/fr3_sharpa_<side>.usda
"""
import os
import sys

from isaacsim import SimulationApp

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.expanduser("~/workspace/fr3-sharpa-rubik/assets"), help="<side>/robot.usd under this directory")
ap.add_argument("--dst", default="assets_franka", help="writes <dst>/<side>/fr3_sharpa_<side>.usda (relative to the project)")
A_ = ap.parse_args()
app = SimulationApp({"headless": True})
from pxr import Sdf, Usd, UsdPhysics  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FR3 = os.path.expanduser("~/workspace/fr3-sharpa-rubik")
ATTRS = ["drive:angular:physics:stiffness", "drive:angular:physics:damping", "drive:angular:physics:maxForce",
         "physxJoint:armature", "physxJoint:jointFriction", "physxJoint:maxJointVelocity"]


def joints(stage, side):
    return {p.GetName(): p for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
            if p.GetTypeName() == "PhysicsRevoluteJoint" and p.GetName().startswith(f"{side}_")}


for side in ("left", "right"):
    official = Usd.Stage.Open(os.path.join(ROOT, f"assets/sharpa_wave_float_base/{side}/{side}_sharpa_wave_with_float_base_with_wrist.usda"))
    src = joints(official, side)
    out_dir = os.path.join(ROOT, A_.dst, side)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"fr3_sharpa_{side}.usda")
    stage = Usd.Stage.CreateNew(out)
    root = stage.DefinePrim(f"/fr3_sharpa_{side}")
    root.GetReferences().AddReference(os.path.realpath(os.path.join(A_.src, f"{side}/robot.usd")))
    stage.SetDefaultPrim(root)
    # the URDF importer nests the colliders in instances: inside one, the spawner can neither author contact offsets
    # nor does the pad material bound on the link reach the collider (PhysX then used 0.5 friction, rigid contact)
    n_inst = 0
    while True:
        inst = [p.GetPath() for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()) if p.IsInstance()]
        if not inst:
            break
        for ip in inst:
            stage.OverridePrim(ip).SetInstanceable(False)
        n_inst += len(inst)
    print(f"@@ {side}: {n_inst} instances made editable", flush=True)
    dst = joints(stage, side)
    n = 0
    for name, sp in src.items():
        if name not in dst:
            continue
        over = stage.OverridePrim(dst[name].GetPath())
        for an in ATTRS:
            a = sp.GetAttribute(an)
            if a and a.Get() is not None:
                t = a.GetTypeName()
                over.CreateAttribute(an, t if t else Sdf.ValueTypeNames.Float).Set(a.Get())
        n += 1
    # the official hand's collision filters (palm / metacarpals vs the proximal phalanges overlap mechanically)
    by_name = {p.GetName(): p.GetPath() for p in root.GetChildren()}  # the links sit directly under the robot prim
    n_f = 0
    for sp in Usd.PrimRange.Stage(official, Usd.TraverseInstanceProxies()):
        if sp.HasAPI(UsdPhysics.FilteredPairsAPI) and sp.GetName() in by_name:
            tg = [by_name[t.name] for t in UsdPhysics.FilteredPairsAPI(sp).GetFilteredPairsRel().GetTargets() if t.name in by_name]
            api = UsdPhysics.FilteredPairsAPI.Apply(stage.OverridePrim(by_name[sp.GetName()]))
            api.CreateFilteredPairsRel().SetTargets(tg)
            n_f += len(tg)
    print(f"@@ {side}: {n_f} collision filter pairs copied", flush=True)
    stage.GetRootLayer().Save()
    # diagnostic variant: arm links and adapter without colliders
    nc = Usd.Stage.CreateNew(out.replace(".usda", "_nocol.usda"))
    r2 = nc.DefinePrim(f"/fr3_sharpa_{side}")
    r2.GetReferences().AddReference(out)
    nc.SetDefaultPrim(r2)
    arm = [f"fr3_link{i}" for i in range(9)] + [f"{side}_adapter"]
    hits = [p for p in Usd.PrimRange.Stage(nc, Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.CollisionAPI) and any(f"/{l}/" in str(p.GetPath()) for l in arm)]
    paths, inst = [h.GetPath() for h in hits], set()
    for h in hits:  # colliders inside instances: de-instance the instance ancestors first
        anc = h.GetParent()
        while anc and anc.GetPath() != Sdf.Path.absoluteRootPath:
            if anc.IsInstance():
                inst.add(anc.GetPath())
            anc = anc.GetParent()
    for ip in inst:
        nc.OverridePrim(ip).SetInstanceable(False)
    for pth in paths:
        nc.OverridePrim(pth).CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    m = len(paths)
    nc.GetRootLayer().Save()
    print(f"@@ {side}: nocol variant, {m} arm/adapter colliders disabled", flush=True)
    missing = sorted(set(src) - set(dst))
    print(f"@@ {side}: {n} hand joints overridden -> {out}; official joints missing here: {missing}", flush=True)
sys.stdout.flush()
os._exit(0)
