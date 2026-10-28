"""Optimize a withdrawal endpoint inside the unchanged inner-ready box.
Offline diagnostic only; independent whole-hand segment audit is mandatory.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch,trimesh
from scipy.optimize import minimize
from rubik_lab.hand_kinematics import HandFK,axis_angle_to_mat
from rubik_lab.diagnostics.convex_geometry import convex_features,convex_overlap_torch,convex_box_overlap_torch


def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if any(a.out.with_suffix(s).exists() for s in ('.json','.npz')):raise FileExistsError(a.out)
 r=json.loads(a.candidate.with_suffix('.json').read_text());z=np.load(a.candidate.with_suffix('.npz'));pr=json.loads(Path(r['source']).with_suffix('.json').read_text());row=pr['results'][0];raw=np.load(Path(pr['source']).with_suffix('.npz'));paths=np.load(Path(r['source']).with_suffix('.npz'));audit=json.loads(a.audit.read_text())
 if audit['source']!=str(a.candidate):raise ValueError('Audit source mismatch')
 pairs=sorted({tuple(c['pair']) for entry in audit['results'] for sample in [dict(collisions=entry['candidate_collisions']),*entry.get('segment_samples',[])] for c in sample['collisions'] if c['rigid_pair'] and c['sat_signed_overlap_m']>.0001})
 if not pairs or any(not x.startswith('right_') for pair in pairs for x in pair):raise ValueError('This diagnostic requires right-hand collision pairs')
 torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
 fk=HandFK('right','cpu',pts_per_link=8,seed=0)
 for table in (fk.T0,fk.axis,fk.link_pts):
  for key in table:table[key]=table[key].double()
 q0=z['initial'][0,:22].copy();x0=z['solution'][0,5:9].copy();ready=raw['ready_targets'][row['trial'],row['env'],1,5:9].astype(float);bias=paths['U_target_bias'][0]
 low=np.maximum.reduce([ready-.1,fk.lower.numpy()[5:9],paths['U_actuator_lower'][0,5:9]-bias[5:9]])
 high=np.minimum.reduce([ready+.1,fk.upper.numpy()[5:9],paths['U_actuator_upper'][0,5:9]-bias[5:9]])
 if (low>high).any():raise ValueError('Empty unchanged ready box')
 wp=torch.tensor(paths['U_wrist_pos'][0:1]);wr=torch.tensor(paths['U_wrist_rot'][0:1]);rot=axis_angle_to_mat(torch.tensor([[0.,0.,np.deg2rad(-row['error_deg'])]]))[0]
 hulls=np.load(Path(r['cooked_hulls']).with_suffix('.npz'));features={}
 links=sorted(set(x for pair in pairs for x in pair)|{x for x in fk.link_pts if x.startswith('right_index_')})
 for link in links:
  mesh=trimesh.Trimesh(vertices=hulls[link+'_vertices'],process=False).convex_hull
  features[link]=tuple(torch.tensor(v) for v in convex_features(mesh.vertices,mesh.faces))
 fractions=[.2,.4,.6,.8,1.]
 def cons(x):
  end=torch.tensor(q0);end[5:9]=torch.tensor(x);values=[]
  for f in fractions:
   q=torch.tensor(q0)+f*(end-torch.tensor(q0));frames=fk.forward(wp,wr,q[None]);depths=[]
   for first,second in pairs:
    ap,ar=frames[first];bp,br=frames[second];depths.append(convex_overlap_torch(features[first],features[second],ar[0],ap[0],br[0],bp[0]))
   for link in links:
    if not link.startswith('right_index_'):continue
    pos,ori=frames[link]
    for body,center,half in [('core',[0,0,-.035/3],[.035,.035,2*.035/3]),('top',[0,0,2*.035/3],[.035,.035,.035/3])]:
     if link.endswith('_elastomer') and body=='top':continue
     R=rot if body=='top' else torch.eye(3)
     depths.append(convex_box_overlap_torch(features[link],ori[0],pos[0],R,R@torch.tensor(center),torch.tensor(half)))
   values.extend(((.00005-torch.stack(depths))/.001).tolist())
  return np.asarray(values)
 objective=lambda x:float(np.sum((x-x0)**2))
 result=minimize(objective,np.clip(x0,low,high),method='SLSQP',bounds=list(zip(low,high)),constraints=[dict(type='ineq',fun=cons)],options=dict(maxiter=150,ftol=1e-10))
 end=q0.copy();end[5:9]=result.x;duration=max(float(np.max(np.abs(end-q0))),.025);margins=cons(result.x)
 r.update(parent_candidate=str(a.candidate),phase='withdrawal_collision_constrained_inner_ready',duration_seconds=duration,predicted_ready_error_rad=float(np.max(np.abs(end[5:9]-ready))),constrained_hand_pairs=pairs,optimization=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),min_constraint=float(margins.min()),fractions=fractions,clearance_limit_m=.00005,endpoint_box_rad=.1),scope='Offline candidate only; no execution or physical completion evidence')
 r.pop('independent_audit',None);r['results'][0].update(measured_feasible=False,segment_pass=False)
 if duration>=r['remaining_original_seconds']:raise ValueError('Exceeds original deadline')
 a.out.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(a.out.with_suffix('.npz'),initial=z['initial'],solution=np.r_[end,np.zeros(6)][None])
 print(json.dumps(dict(optimization=r['optimization'],duration_seconds=duration,predicted_ready_error_rad=r['predicted_ready_error_rad'])),flush=True)

if __name__=='__main__':main()
