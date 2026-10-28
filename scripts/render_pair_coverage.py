"""Fixed pair-coverage comparison; includes all four tested U/Up pairs."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
if out.with_suffix('.png').exists() or out.with_suffix('.json').exists():raise FileExistsError(out)
names=[['pair_U_U_v1','sequence_Up200_verify'],['pair_Up_U_v1','pair_Up_Up_v1']]
values=np.zeros((2,2));labels={};sources=[]
for i in range(2):
 for j in range(2):
  path=Path('outputs/primitive_cycle')/(names[i][j]+'.json');raw=path.read_bytes();r=json.loads(raw);s=r['summary'];values[i,j]=100*s['success_rate'];st=r['per_primitive_stage']
  labels[i,j]=f"{values[i,j]:.1f}%\n{s['reasons'].get('success',0)}/{s['n']} complete trials\nSecond move: {st[1]['completed']}/{st[1]['attempted']}"
  sources.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'summary':s,'per_stage':st})
fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(values,vmin=0,vmax=100,cmap='Blues')
for (i,j),label in labels.items():ax.text(j,i,label,ha='center',va='center',fontsize=13,color='white' if values[i,j]>60 else 'black')
ax.set_xticks([0,1],['U',"U'"]);ax.set_yticks([0,1],['U',"U'"]);ax.set_xlabel('Second commanded move');ax.set_ylabel('First commanded move')
ax.set_title('Primitive pair coverage: ground-truth switching',pad=16,fontsize=13)
fig.colorbar(im,ax=ax,label='Complete trials (%)',shrink=.7)
fig.text(.5,.025,'Original U + Up200; selected canonical grasp; no reset inside each pair.\nDifferent fixed evaluation seeds and trial quotas. These are not tactile-autonomy results.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.09,1,.95]);fig.savefig(out.with_suffix('.png'),dpi=160,bbox_inches='tight')
out.with_suffix('.json').write_text(json.dumps({'scope':'All four same-face pairs, GT diagnostic only.','sources':sources},indent=2)+'\n')
