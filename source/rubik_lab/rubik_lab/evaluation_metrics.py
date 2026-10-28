"""Sequence metrics distinguish total reliability from conditional skill reliability."""
from statistics import median

def summarize_stages(records,sequence):
    if not sequence:return None
    result=[]
    for stage,move in enumerate(sequence):
        attempted=[r for r in records if len(r.get('primitives',[]))>=stage]
        completed=[r for r in attempted if len(r.get('primitives',[]))>stage]
        failed=[r for r in attempted if len(r.get('primitives',[]))==stage]
        result.append({'stage':stage,'move':move,'attempted':len(attempted),'completed':len(completed),
            'conditional_completion_rate':len(completed)/len(attempted) if attempted else None,
            'completion_rate_over_all_trials':len(completed)/len(records) if records else None,
            'premature_requests':sum(r.get('reason')=='premature_trigger' for r in failed),
            'median_seconds_completed':median(r['primitives'][stage]['seconds'] for r in completed) if completed else None})
    return result


def summarize_mechanical(report):
    """Roundwise descriptive statistics; not a blind-policy acceptance verdict."""
    from collections import Counter
    from math import sqrt
    rows=report['episodes']; rounds=report['rounds']; environments=report['num_envs']
    if rounds<=0 or environments<=0 or environments%2:raise ValueError('Expected positive rounds and an even mechanical environment count')
    ids=[(r['trial'],r['env']) for r in rows]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate mechanical trial/environment records')
    expected={(t,e) for t in range(rounds) for e in range(environments)}
    if any(r['move'] not in ('U',"U'") for r in rows): raise ValueError('Unexpected mechanical move')
    def group(items):
        n=len(items); k=sum(r['reason']=='success' for r in items)
        interval=None
        if n:
            z=1.959963984540054; p=k/n; den=1+z*z/n
            center=(p+z*z/(2*n))/den
            half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
            interval=[max(0.,center-half),min(1.,center+half)]
        return dict(n=n,success=k,success_rate=k/n if n else None,wilson95=interval,
                    failure_types=dict(Counter(r['reason'] for r in items if r['reason']!='success')))
    return dict(quota_complete=set(ids)==expected and all(r["move"]==('U' if r["env"]<environments//2 else "U'") for r in rows),expected_trials=len(expected),observed_trials=len(rows),
                by_move={m:dict(expected_trials=rounds*environments//2,expected_per_round=environments//2,overall=group([r for r in rows if r['move']==m]),
                    rounds=[dict(round=t,**group([r for r in rows if r['move']==m and r['trial']==t])) for t in range(rounds)]) for m in ('U',"U'")},
                premature_confirmation_rate=None,
                limitation='Privileged mechanical completion has no blind confirmation requests. Wilson intervals are descriptive binomial intervals; repeated canonical grasp/sensor seeds do not establish independent physical generalization.')


def summarize_continuation(report):
    """Frozen four-chain quota with unconditional and second-stage reliability."""
    from collections import Counter
    from math import sqrt
    pairs=[('U','U'),('U',"U'"),("U'",'U'),("U'","U'")]
    nenv,rounds=report['num_envs'],report['rounds']
    if nenv<=0 or nenv%4 or rounds<=0:raise ValueError('Expected four equal groups and positive rounds')
    rows=report['episodes'];ids=[(r['trial'],r['env']) for r in rows]
    if len(set(ids))!=len(ids) or set(ids)!={(t,e) for t in range(rounds) for e in range(nenv)}:
        raise ValueError('Incomplete or duplicate continuation quota')
    for r in rows:
        if tuple(r['pair'])!=pairs[r['env']//(nenv//4)]:raise ValueError('Wrong chain assignment')
        if r['completed_stages'] not in (0,1,2) or (r['completed_stages']==2)!=(r['reason']=='success'):
            raise ValueError('Inconsistent completion record')
    def interval(k,n):
        if not n:return None
        z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
        d=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return [max(0.,c-d),min(1.,c+d)]
    def group(items):
        n=len(items);first=sum(r['completed_stages']>=1 for r in items);k=sum(r['reason']=='success' for r in items)
        return {'n':n,'first_complete':first,'two_complete':k,'success_rate':k/n,
                'wilson95':interval(k,n),'second_conditional_rate':k/first if first else None,
                'second_conditional_wilson95':interval(k,first),
                'failure_types':dict(Counter(r['reason'] for r in items if r['reason']!='success'))}
    return {'quota_complete':True,'chains':{'->'.join(p):{'overall':group([r for r in rows if tuple(r['pair'])==p]),
        'rounds':[{'round':t,**group([r for r in rows if tuple(r['pair'])==p and r['trial']==t])} for t in range(rounds)]} for p in pairs},
        'premature_confirmation_rate':None,
        'limitation':'Privileged completion, not blind acceptance. Wilson intervals are descriptive; repeated canonical grasp seeds do not establish independent physical generalization.'}


def summarize_withdrawal(report, raw):
    """Actual return readiness from recorded sensors; phase labels are not gates.

    Only second U in UU is modified by the withdrawal diagnostic. Signed angle
    errors must be absolutized. Four thresholds below are the frozen contract.
    """
    import numpy as np
    from rubik_lab.constants import FINGER_JOINTS
    cols = np.array([i for i, name in enumerate(FINGER_JOINTS)
                     if name.startswith("index")][:4])
    trace, sensory = raw["trace"], raw["sensory"]
    stages = raw["withdrawal_stage"]
    rows = []
    for episode in report["episodes"]:
        if episode["pair"] != ["U", "U"] or episode["completed_stages"] < 1:
            continue
        env = episode["env"]
        mask = ((trace[:, env, 0] == episode["trial"])
                & (trace[:, env, 3] == 1) & (trace[:, env, 4] > 0))
        t = trace[mask, env]
        phase = stages[mask, env]
        speed = np.abs(sensory[mask, env][:, 66 + cols]).max(axis=-1)
        fingers_ready = (t[:, 7] <= .2) & (speed <= 1.) & (t[:, 8] <= .5)
        returning = phase >= 2
        near_slow = (np.abs(t[:, 5]) < 5.) & (np.abs(t[:, 6]) < .25)
        ready_at = np.flatnonzero(fingers_ready & returning)
        return_at = np.flatnonzero(returning)
        release_at = np.flatnonzero(phase > 0)
        release = int(release_at[0]) if len(release_at) else None
        if report.get("hold_then_withdraw") and release is not None:
            if release < 4 or not ((np.abs(t[release-4:release,5]) < report.get("release_angle_tolerance_deg",3.))
                                   & (np.abs(t[release-4:release,6]) < .25)).all():
                raise ValueError("Withdrawal did not follow four valid settled samples")
        k = int(ready_at[0]) if len(ready_at) else None
        # Renewed contact is measured after the clearance transition, not during
        # the intended push/brake phase. A contact frame is not an impulse.
        rows.append({"trial": episode["trial"], "env": env,
                     "success": episode["reason"] == "success",
                     "release_step": release,
                     "angle_at_release_deg": float(t[release-1,5]) if release is not None and release > 0 else None,
                     "peak_speed_after_release": float(np.abs(t[release:,6]).max()) if release is not None else None,
                     "final_angle_error_deg": float(t[-1,5]),
                     "first_return_step": int(return_at[0]) if len(return_at) else None,
                     "first_actual_ready_step": k,
                     "angle_at_actual_ready_deg": float(t[k, 5]) if k is not None else None,
                     "near_slow_frames": int(near_slow.sum()),
                     "near_slow_actual_ready_frames": int((near_slow & fingers_ready & returning).sum()),
                     "renewed_contact_frames": int(((t[:, 8] > .5) & returning).sum()),
                     "instant_full_ready_frames": int(t[:, 9].sum())})
    return {"scope": "Privileged mechanical diagnosis, not blind acceptance; 40Hz samples do not resolve impulses.",
            "stage_definition": "Stage2/3 denotes return in progress, not physical readiness.",
            "rows": rows}


def summarize_terminal_hold(report, raw):
    """Audit the causal hold intervention before interpreting its mechanics."""
    import numpy as np
    result=[]
    for episode in report["episodes"]:
        if episode["pair"] != ["U", "U"] or episode["completed_stages"] < 1:
            continue
        env=episode["env"]
        trace=raw["trace"]
        mask=((trace[:,env,0]==episode["trial"]) & (trace[:,env,3]==1)
              & (trace[:,env,4]>0))
        t=trace[mask,env]
        h=raw["terminal_hold"][mask,env]
        indices=np.flatnonzero(h)
        if not len(indices):
            continue
        k=int(indices[0])
        if k==0:
            raise ValueError("Hold triggered at handoff before second-action progress")
        if abs(t[k-1,5])>=15 or abs(t[k-1,6])>=1:
            raise ValueError("Hold trigger does not match current pre-action state")
        commands=raw["commands"][mask,env]
        if not np.array_equal(commands[k:],np.broadcast_to(commands[k-1],commands[k:].shape)):
            raise ValueError("Hold must preserve the last issued command")
        end=min(k+8,len(t))
        result.append({"trial":episode["trial"],"env":env,"reason":episode["reason"],
                       "latch_step":k,"initial_error_deg":float(t[k-1,5]),
                       "peak_speed_200ms":float(np.abs(t[k:end,6]).max()),
                       "peak_speed_after_latch":float(np.abs(t[k:,6]).max()),
                       "angle_change_200ms_deg":float(t[end-1,5]-t[k-1,5]),
                       "final_error_deg":float(t[-1,5])})
    return {"triggered":len(result),"command_hold_verified":True,"rows":result,
            "secondary_speed_over1":sum(row["peak_speed_200ms"]>1 for row in result),
            "median_peak_200ms":float(np.median([row["peak_speed_200ms"] for row in result])) if result else None}


def summarize_release_substeps(report, raw):
    """Compare joint position increments with three velocity integration rules.

    Position correction can differ from velocity integration in a constrained
    solver; this is a consistency diagnostic, not an energy/conservation proof.
    """
    import numpy as np
    values=raw.get("release_substeps")
    if values is None or len(values)==0:
        return {"n_substep_rows":0,"segments":[],"max_abs_integral_residual_deg":None}
    if values.ndim!=2 or values.shape[1]!=7 or not np.isfinite(values).all():
        raise ValueError("Malformed release substep record")
    dt=1/float(report["physics_hz"])
    rows=[]
    for trial,env in np.unique(values[:,[0,2]],axis=0):
        z=values[(values[:,0]==trial)&(values[:,2]==env)]
        if len(z)<2 or not (np.diff(z[:,1])==1).all():
            raise ValueError("Substep segment must contain consecutive physical steps")
        q,w=z[:,3],z[:,4]
        delta=float(q[-1]-q[0])
        integrals=np.array([w[:-1].sum(),w[1:].sum(),((w[:-1]+w[1:])*.5).sum()])*dt
        rows.append({"trial":int(trial),"env":int(env),"n":len(z),
                     "duration_s":(len(z)-1)*dt,"top_force_max":float(z[:,5].max()),
                     "angle_change_deg":float(np.rad2deg(delta)),
                     "velocity_integrals_deg":np.rad2deg(integrals).tolist(),
                     "integral_residuals_deg":np.rad2deg(delta-integrals).tolist(),
                     "sign_flip_fraction":float(np.mean(w[:-1]*w[1:]<0)),
                     "speed_minmax":[float(w.min()),float(w.max())]})
    return {"n_substep_rows":len(values),"segments":rows,
            "integration_order":["pre","post","trapezoid"],
            "max_abs_integral_residual_deg":max(abs(v) for row in rows for v in row["integral_residuals_deg"])}


def audit_recontact_completion(report, raw):
    """Recompute terminal gates from raw measurements for successful dual-index runs.

    Independent NumPy readout, not a policy input. Whole-sequence motion still
    uses recorded peak measurements; this is not hardware validation or a
    physics-substep audit. Missing measurements yield unknown, never a pass.
    """
    import numpy as np
    required=('trace','recontact','sensory','contact','poses','ready_targets','core_angular_speed')
    missing=[key for key in required if key not in raw]
    if missing:return dict(status='unknown',missing=missing,rows=[])
    rows=[]
    for episode in report['episodes']:
        if episode['reason']!='success':continue
        e,t=episode['env'],episode['trial']
        index=np.flatnonzero((raw['trace'][:,e,0]==t)&(raw['trace'][:,e,3]==1)&(raw['trace'][:,e,4]>0))
        if not len(index) or not (raw['recontact'][index,e,2]>0).any():continue
        window=index[-10:];trace=raw['trace'][window,e];sensory=raw['sensory'][window,e]
        forces=raw['contact'][window,e];pose=raw['poses'][window,e]
        core=raw['core_angular_speed'][window,e]
        ready=raw['ready_targets'][t,e]
        fingers=[]
        for h in (0,1):
            q=sensory[:,h*22+5:h*22+9];qd=sensory[:,44+h*22+5:44+h*22+9]
            fingers.append(dict(error=float(np.abs(q-ready[h,5:9]).max()),
                speed=float(np.abs(qd).max()),release=float(forces[:,1,h,6:11].sum(-1).max())))
        support=forces.sum(1)[:,:,[5,10,15,20,26]]
        pad_count=(support>.05).sum(-1)
        quat=pose[:,3:7];quat=quat/np.linalg.norm(quat,axis=-1,keepdims=True)
        rot=2*np.arccos(np.clip(np.abs(quat@quat[0]),0,1))
        pos=np.linalg.norm(pose[:,:3]-pose[0,:3],axis=-1)
        checks=dict(deadline=bool((raw['trace'][index[-1],e,1]-raw['trace'][index[0],e,1]+1)/report.get('control_hz',40)<=6),ten_consecutive_frames=len(window)==10 and bool(np.all(np.diff(trace[:,1])==1)),
            finite=all(np.isfinite(x).all() for x in (trace,sensory,forces,pose,core,ready)),
            angle=bool((np.abs(trace[:,5])<5).all()),layer_speed=bool((np.abs(trace[:,6])<.25).all()),
            core_speed=bool((core<1).all()),fingers=all(f['error']<=.2 and f['speed']<=1 and f['release']<=.5 for f in fingers),
            support=bool((pad_count>=1).all()),pose_dwell=bool((pos<=.002).all() and (rot<=np.deg2rad(2)).all()),
            sequence_motion=bool((trace[:,12]<=.015).all() and (trace[:,13]<=np.deg2rad(15)).all() and (trace[:,14]<=.01).all() and (trace[:,15]<=np.deg2rad(10)).all()),
            no_failure=bool((trace[:,11]==0).all()))
        rows.append(dict(trial=t,env=e,checks=checks,passed=all(checks.values()),fingers=fingers,
            angle_max_deg=float(np.abs(trace[:,5]).max()),core_speed_max=float(core.max()),
            dwell_pos_max_m=float(pos.max()),dwell_rotation_max_deg=float(np.rad2deg(rot).max())))
    return dict(status='checked',rows=rows,limitation='Control-frame measurements and recorded sequence peaks; not independent hardware or substep evidence. No claim follows from an empty rows list.')
