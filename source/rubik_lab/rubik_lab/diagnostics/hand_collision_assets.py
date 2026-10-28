"""Export actual PhysX cooked hand hulls without stepping or changing source assets."""
import argparse,json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
if a.out.with_suffix('.json').exists() or a.out.with_suffix('.npz').exists():raise FileExistsError(a.out)
app=AppLauncher(a).app
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,PhysicsSchemaTools,PhysxSchema,Gf
from omni.physx import get_physx_cooking_interface
from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
from rubik_lab.constants import ASSET_DIR
cache=UsdUtils.StageCache.Get();cooking=get_physx_cooking_interface()
def cook(stage,prim):
    sid=cache.Insert(stage).ToLongInt();result={}
    def receive(status,hulls):
        result['status']=str(status)
        if status!=PhysxCollisionRepresentationResult.RESULT_VALID:raise RuntimeError(str(status))
        result['vertices']=[np.array([[v.x,v.y,v.z] for v in h.vertices],dtype=np.float64) for h in hulls]
    cooking.request_convex_collision_representation(stage_id=sid,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),run_asynchronously=False,on_result=receive)
    if 'vertices' not in result:raise RuntimeError('No synchronous cooking result: '+str(prim.GetPath()))
    return result
# Verify coordinate interpretation on a translated, known-size mesh.
probe=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(probe,1.)
m=UsdGeom.Mesh.Define(probe,'/probe');points=[[-.01,-.02,-.03],[.01,-.02,-.03],[.01,.02,-.03],[-.01,.02,-.03],[-.01,-.02,.03],[.01,-.02,.03],[.01,.02,.03],[-.01,.02,.03]]
m.CreatePointsAttr(points);m.CreateFaceVertexCountsAttr([4]*6);m.CreateFaceVertexIndicesAttr([0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7]);m.AddTranslateOp().Set(Gf.Vec3d(.1,.2,.3))
UsdPhysics.CollisionAPI.Apply(m.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
probe_result=cook(probe,m.GetPrim());v=np.concatenate(probe_result['vertices'])
if not (np.allclose(v.min(0),[-.01,-.02,-.03],atol=1e-6) and np.allclose(v.max(0),[.01,.02,.03],atol=1e-6)):raise AssertionError('Cooked vertices are not expected mesh-local coordinates')
rows=[];raw={}
for side in ('left','right'):
    path=Path(ASSET_DIR)/f'sharpa_wave_float_base/{side}/{side}_sharpa_wave_with_float_base_with_wrist.usda'
    stage=Usd.Stage.Open(str(path));xforms=UsdGeom.XformCache()
    for prim in stage.TraverseAll():
        if prim.GetTypeName()!='Mesh' or not prim.GetAttribute('physics:collisionEnabled').Get():continue
        link=str(prim.GetPath()).split('/')[2];linkprim=stage.GetPrimAtPath('/'+str(prim.GetPath()).split('/')[1]+'/'+link)
        result=cook(stage,prim)
        if len(result['vertices'])!=1:raise ValueError('Expected one convex hull per collider')
        transform=np.asarray(xforms.ComputeRelativeTransform(prim,linkprim)[0]);vertices=result['vertices'][0]
        raw[link+'_vertices']=vertices@transform[:3,:3]+transform[3,:3]
        api=PhysxSchema.PhysxConvexHullCollisionAPI(prim)
        row=dict(side=side,link=link,path=str(prim.GetPath()),asset=str(path),vertices=len(vertices),hull_vertex_limit=api.GetHullVertexLimitAttr().Get(),minimum_thickness=api.GetMinThicknessAttr().Get(),status=result['status'])
        rows.append(row);print(json.dumps(row),flush=True)
a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out.with_suffix('.npz'),**raw)
a.out.with_suffix('.json').write_text(json.dumps(dict(probe_mesh_local_coordinate_check=True,scope='PhysX cooking API output for original USD assets; vertices stored in each hand link frame. No source asset modification, simulation stepping, or task acceptance.',colliders=rows),indent=2)+'\n')
app.close()
