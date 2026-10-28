"""Tactile readiness inference and a consecutive-sample completion gate."""
import torch
from .model import CompletionDetector, AngleCompletionDetector
from .data import N_MOVES


class StreamingReadiness:
    def __init__(self,checkpoint,n,device,threshold=0.95):
        if checkpoint.get('format') not in ('streaming_tactile_ready_v1','streaming_tactile_ready_angle_v1','streaming_tactile_ready_normalized_v1','streaming_tactile_ready_angle_normalized_v1'):
            raise ValueError('Expected whole-history instantaneous-readiness detector')
        self.angle_gate_deg=checkpoint.get('angle_gate_deg')
        ft_dim=checkpoint.get("ft_dim",60)
        self.model=(AngleCompletionDetector(ft_dim=ft_dim) if self.angle_gate_deg is not None else CompletionDetector(ft_dim=ft_dim)).to(device)
        self.wrench_rotation=None
        if checkpoint.get("kinematic_ft",False):
            from .kinematic_features import PadWrenchRotation
            self.wrench_rotation=PadWrenchRotation().to(device)
            if self.wrench_rotation.urdf_sha256!=checkpoint.get("kinematics_urdf_sha256"):
                raise ValueError("Hand geometry differs from detector training geometry")
        if ft_dim!=(120 if self.wrench_rotation is not None else 60):
            raise ValueError("Unexpected tactile feature layout")
        self.model.load_state_dict(checkpoint['model'])
        self.model.set_normalization(checkpoint.get('feature_normalization'))
        self.model.eval()
        self.hidden=torch.zeros(1,n,self.model.gru.hidden_size,device=device)
        self.stable_steps=torch.zeros(n,dtype=torch.long,device=device)
        self.label_key=checkpoint.get('label_key','cycle_ready')
        if self.label_key not in ('cycle_ready','primitive_done'):raise ValueError('Unknown readiness label semantics')
        self.hold_steps=int(checkpoint['hold_steps'])
        if self.hold_steps<1:raise ValueError('Completion dwell must be positive')
        if not 0<threshold<=1:raise ValueError('Readiness threshold must be in (0,1]')
        self.threshold=threshold
        self.preserve_history_between_moves=bool(checkpoint.get("preserve_history_between_moves",False))

    @torch.inference_mode()
    def reset(self,ids=None):
        if ids is None:
            self.hidden.zero_()
            self.stable_steps.zero_()
        else:
            self.hidden[:,ids]=0
            self.stable_steps[ids]=0

    @torch.inference_mode()
    def advance(self,ids):
        if self.preserve_history_between_moves:
            self.stable_steps[ids]=0
        else:
            self.reset(ids)

    @torch.inference_mode()
    def step(self,*,tactile_ft,finger_pos,finger_tgt,move_id):
        prop=torch.cat([finger_pos.flatten(1),(finger_tgt-finger_pos).flatten(1)],-1)
        ft=tactile_ft.flatten(1)
        if self.wrench_rotation is not None:
            ft=torch.cat([ft,self.wrench_rotation(tactile_ft,finger_pos).flatten(1)],-1)
        x={'ft':ft.unsqueeze(1)*0.2,'prop':prop.unsqueeze(1),
           'move':torch.nn.functional.one_hot(move_id.long(),N_MOVES).float()}
        angle=None
        if self.angle_gate_deg is not None:
            logits,angle,self.hidden=self.model.predict_sequence(x,self.hidden)
        else:
            logits,self.hidden=self.model.forward_sequence(x,self.hidden)
        probabilities=logits[:,0].softmax(-1)
        ready=probabilities[:,1]>=self.threshold
        if angle is not None:
            ready &= torch.rad2deg(angle[:,0]).abs()<=self.angle_gate_deg
        self.stable_steps=torch.where(ready,self.stable_steps+1,0)
        result={'ready_probability':probabilities[:,1],'failed_probability':probabilities[:,2],
                'done':self.stable_steps>=self.hold_steps,'stable_steps':self.stable_steps.clone()}
        if angle is not None:
            result['angle_error_deg']=torch.rad2deg(angle[:,0])
        return result


class PerMoveReadiness:
    """Route per-primitive observers by the commanded move, without cube state.

    Each boundary clears both observers for affected environments. Continuous
    observers and angle-head mixtures are rejected to keep history semantics explicit.
    """
    def __init__(self, observers):
        self.observers=dict(observers)
        if not self.observers:raise ValueError("Need at least one observer")
        if any(o.preserve_history_between_moves or o.angle_gate_deg is not None for o in self.observers.values()):
            raise ValueError("Per-move routing requires plain per-primitive observers")
        if len({o.hold_steps for o in self.observers.values()})!=1:
            raise ValueError("All observers must use the same completion dwell")
        if len({o.label_key for o in self.observers.values()})!=1:
            raise ValueError("Per-move observers must share readiness label semantics")
        self.label_key=next(iter(self.observers.values())).label_key
        self.preserve_history_between_moves=False
        self.angle_gate_deg=None

    @property
    def hold_steps(self):return next(iter(self.observers.values())).hold_steps

    @hold_steps.setter
    def hold_steps(self,value):
        for observer in self.observers.values():observer.hold_steps=value

    def reset(self,ids=None):
        for observer in self.observers.values():observer.reset(ids)

    def advance(self,ids):self.reset(ids)

    @torch.inference_mode()
    def step(self,**inputs):
        move=inputs["move_id"]
        known=torch.zeros_like(move,dtype=torch.bool)
        result=None
        for mid,observer in self.observers.items():
            selected=move==mid
            known |= selected
            prediction=observer.step(**inputs)
            if result is None:result={key:torch.zeros_like(value) for key,value in prediction.items()}
            for key,value in prediction.items():result[key]=torch.where(selected,value,result[key])
        if not known.all():raise ValueError("No readiness observer for a commanded move")
        return result
