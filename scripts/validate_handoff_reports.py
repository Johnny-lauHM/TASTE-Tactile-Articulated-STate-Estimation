"""Verify final inventory against immutable raw episode records and report hashes."""
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('inventory');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
if out.exists():raise FileExistsError(out)
raw=Path(a.inventory).read_bytes();inventory=json.loads(raw);checks=[]
for group,entries in inventory['groups'].items():
 for entry in entries:
  if entry['status']!='reported':raise ValueError('Unfinished inventory entry: '+entry['name'])
  path=Path(entry['path']);content=path.read_bytes()
  assert hashlib.sha256(content).hexdigest()==entry['sha256'],str(path)+' hash'
  r=json.loads(content);episodes=r['episodes'];s=r['summary'];n=len(episodes)
  assert n==s['n'] and n>0,str(path)+' quota'
  assert dict(Counter(e['reason'] for e in episodes))==s['reasons'],str(path)+' reasons'
  assert all(bool(e['success'])==(e['reason']=='success') for e in episodes),str(path)+' success labels'
  assert abs(sum(e['success'] for e in episodes)/n-s['success_rate'])<1e-12,str(path)+' success rate'
  ids={(e['trial'],e['env']) for e in episodes};rounds=sorted({t for t,e in ids});envs=r['config']['scene']['num_envs']
  assert len(ids)==n and rounds==list(range(len(rounds))),str(path)+' identities'
  assert ids=={(t,e) for t in rounds for e in range(envs)},str(path)+' missing env trial'
  for t in rounds:
   rows=[e for e in episodes if e['trial']==t];rs=r['per_trial'][str(t)]
   assert rs['n']==len(rows) and abs(rs['success_rate']-sum(e['success'] for e in rows)/len(rows))<1e-12,str(path)+' round'
  for stage in r.get('per_primitive_stage',[]):
   k=stage['stage'];completed=sum(len(e.get('primitives',[]))>k for e in episodes)
   attempted=n if k==0 else sum(len(e.get('primitives',[]))>=k for e in episodes)
   assert completed==stage['completed'] and attempted==stage['attempted'],str(path)+' stage'
  for e in episodes:
   if not e['success']:continue
   assert len(e['primitives'])==len(r['sequence']),str(path)+' completed chain'
   assert all(abs(x['error_deg'])<=5.00001 and x['seconds']<=6.00000001 for x in e['primitives']),str(path)+' primitive limits'
   assert e['peak_wrist_mm']<=10.00001 and e['peak_wrist_deg']<=10.00001 and e['peak_cube_mm']<=15.00001 and e['peak_cube_deg']<=15.00001,str(path)+' whole chain limits'
  checks.append({'group':group,'name':entry['name'],'trials':n,'rounds':len(rounds),'passed':True})
result={'inventory':a.inventory,'inventory_sha256':hashlib.sha256(raw).hexdigest(),'scope':'Artifact consistency, fixed quotas, reported stages and successful-record angle/time/motion limits. This does not certify sensor realism or autonomous recovery.','passed':True,'reports':checks}
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'reports':len(checks),'trials':sum(x['trials'] for x in checks),'passed':True}))
