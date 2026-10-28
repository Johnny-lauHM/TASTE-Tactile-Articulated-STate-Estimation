"""Rotate pad-frame wrench vectors into wrist-aligned axes using finger encoders.

Moments remain about each pad origin: this is an axis rotation, not a wrench
translation to the wrist. No cube state, contact identity, or vision is used.
The delivered wrench is paired with current encoder samples; sensor age remains
part of the learned history, matching the existing raw dataset timing.
"""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from torch import nn
from rubik_lab.constants import ASSET_DIR,FINGER_JOINTS
from rubik_lab.hand_kinematics import _origin,axis_angle_to_mat


class PadWrenchRotation(nn.Module):
    def __init__(self):
        super().__init__()
        self.chains=[]
        self.urdf_sha256={}
        for side in ('left','right'):
            path=Path(ASSET_DIR)/'sharpa_wave_float_base'/side/(side+'_sharpa_wave_with_float_base_with_wrist.urdf')
            self.urdf_sha256[side]=hashlib.sha256(path.read_bytes()).hexdigest()
            root=ET.parse(path).getroot()
            joints={j.find('child').get('link'):j for j in root.findall('joint')}
            names=[side+'_'+name for name in FINGER_JOINTS]
            for finger in ('thumb','index','middle','ring','pinky'):
                chain=[];link=side+'_'+finger+'_elastomer'
                while link!=side+'_hand_wrist':
                    joint=joints[link]
                    rotation=torch.tensor(_origin(joint.find('origin'))[:3,:3],dtype=torch.float32)
                    index=names.index(joint.get('name')) if joint.get('type')=='revolute' else -1
                    axis=torch.tensor([float(v) for v in joint.find('axis').get('xyz').split()],dtype=torch.float32) if index>=0 else torch.zeros(3)
                    chain.append((rotation,axis,index))
                    link=joint.find('parent').get('link')
                chain.reverse()
                slot=len(self.chains)
                self.register_buffer('origins_'+str(slot),torch.stack([x[0] for x in chain]))
                self.register_buffer('axes_'+str(slot),torch.stack([x[1] for x in chain]))
                self.chains.append([x[2] for x in chain])

    def forward(self,wrench,joints):
        if wrench.shape[-3:]!=(2,5,6) or joints.shape[-2:]!=(2,22) or wrench.shape[:-3]!=joints.shape[:-2]:
            raise ValueError('Expected matching leading dimensions, wrench[...,2,5,6], joints[...,2,22]')
        leading=wrench.shape[:-3]
        w=wrench.reshape(-1,2,5,6);q=joints.reshape(-1,2,22)
        outputs=[]
        for slot,indices in enumerate(self.chains):
            side,finger=divmod(slot,5)
            origins=getattr(self,'origins_'+str(slot));axes=getattr(self,'axes_'+str(slot))
            rotation=torch.eye(3,device=q.device,dtype=q.dtype).expand(len(q),3,3)
            for level,index in enumerate(indices):
                rotation=rotation@origins[level]
                if index>=0:rotation=rotation@axis_angle_to_mat(q[:,side,index,None]*axes[level])
            force=(rotation@w[:,side,finger,:3,None]).squeeze(-1)
            moment=(rotation@w[:,side,finger,3:,None]).squeeze(-1)
            outputs.append(torch.cat([force,moment],-1))
        return torch.stack(outputs,1).reshape(*leading,2,5,6)
