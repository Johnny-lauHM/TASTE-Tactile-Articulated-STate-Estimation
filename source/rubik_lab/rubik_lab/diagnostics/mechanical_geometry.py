"""Offline collision-geometry reconstruction of recorded mechanical states.
Not a live simulation recording, and never an acceptance video.
"""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rubik_lab.hand_kinematics import HandFK,quat_to_mat,_origin,PROJECT_DIR
from rubik_lab.constants import ASSET_DIR
p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--env',type=int,required=True)
p.add_argument('--steps',type=int,nargs='+',required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
z=np.load(a.input.with_suffix('.npz')); env=a.env
if len(np.unique(z['trace'][:,env,0]))!=1:raise ValueError('Select a single-round source for unambiguous step IDs')
side='left' if env>=z['trace'].shape[1]//2 else 'right'; hand=0 if side=='left' else 1
fk=HandFK(side,'cpu',pts_per_link=8)
root=ET.parse(Path(ASSET_DIR)/f'sharpa_wave_float_base/{side}/{side}_sharpa_wave_with_float_base_with_wrist.urdf').getroot()
meshes={}
for link in root.findall('link'):
    name=link.get('name');col=link.find('collision')
    if col is None or not name.startswith(side+'_index_'):continue
    file=col.find('geometry/mesh').get('filename').split('/')[-1]
    mesh=trimesh.load(Path(PROJECT_DIR)/f'sharpa_wave/{side}_sharpa_wave/meshes'/file,force='mesh')
    mesh.apply_transform(_origin(col.find('origin')));meshes[name]=mesh
fig=plt.figure(figsize=(5*len(a.steps),5.4))
for k,step in enumerate(a.steps):
    ids=np.flatnonzero((z['trace'][:,env,1]==step)&(z['trace'][:,env,2]>0))
    if len(ids)!=1:raise ValueError('Frame must be unique and active')
    i=int(ids[0]);s=z['states'][i,env];t=z['trace'][i,env]
    center=s[:3]+z['env_origins'][env]
    wrist=torch.tensor(s[7+hand*7:10+hand*7][None]);rot=quat_to_mat(torch.tensor(s[10+hand*7:14+hand*7][None]))
    q=torch.tensor(s[21+hand*22:43+hand*22][None]);frames=fk.forward(wrist,rot,q)
    ax=fig.add_subplot(1,len(a.steps),k+1,projection='3d')
    triangles=[];colors=[]
    R=quat_to_mat(torch.tensor(s[3:7])).numpy(); angle=np.deg2rad((90 if side=='left' else -90)-t[4])
    Rz=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    for height,zc,orientation,color in [(0.07*2/3,-0.07/6,R,'#aab4bd'),(0.07/3,0.07/3,R@Rz,'#5ba7dc')]:
        box=trimesh.creation.box(extents=[0.07,0.07,height]);vertices=(box.vertices+np.array([0,0,zc]))@orientation.T*1000
        triangles.extend(vertices[box.faces]);colors.extend([to_rgba(color,.24)]*len(box.faces))
    for name,mesh in meshes.items():
        pos,orientation=frames[name];vertices=(mesh.vertices@orientation[0].numpy().T+pos[0].numpy()-center)*1000
        color='#e55353' if name.endswith('_DP') else ('#e8b949' if name.endswith('_elastomer') else '#8495a7')
        triangles.extend(vertices[mesh.faces]);colors.extend([to_rgba(color,1)]*len(mesh.faces))
    ax.add_collection3d(Poly3DCollection(triangles,facecolors=colors,edgecolor='none',zsort='average'))
    force=z['contacts'][i,env,1,hand,6:11].sum()
    ax.set(xlim=(-55,95),ylim=(-75,75),zlim=(-40,80),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
    ax.set_box_aspect((150,150,120));ax.view_init(elev=22,azim=125 if side=='left' else -125)
    ax.set_title(f'Step {step} | error {t[4]:.2f} deg\nIndex top contact {force:.2f} N')
fig.suptitle('Offline collision geometry reconstruction — not live video\nBlue: rotating layer | red: distal link | gold: tactile pad',fontsize=12)
fig.tight_layout(rect=(0,0,1,.9));a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=150);plt.close(fig)
print(str(a.out))
