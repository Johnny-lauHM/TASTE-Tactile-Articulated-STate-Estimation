"""Physical passivity properties, independent of the simulation runtime."""
import importlib.util
from pathlib import Path
import pytest
import torch
spec=importlib.util.spec_from_file_location("passive_friction",Path(__file__).resolve().parents[1]/"source/rubik_lab/rubik_lab/passive_friction.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_spin_down_does_not_gain_energy_or_reverse_for_larger_true_inertia():
    initial=torch.tensor([-10.,-1.,-.1,-.001,0.,.001,.1,1.,10.],dtype=torch.float64)
    for inertia in [1e-4,1.5e-4,1e-3]:
        speed=initial.clone()
        for _ in range(480):
            torque=m.bounded_friction_torque(speed,.01,.05,1/240,torch.full_like(speed,1e-4))
            next_speed=speed+torque/inertia/240
            assert torch.all(next_speed.square()<=speed.square()+1e-12)
            assert torch.all(next_speed*speed>=-1e-12)
            speed=next_speed
        assert speed.abs().max()<1e-6

def test_high_speed_resistance_is_preserved_and_rest_has_no_impulse():
    speed=torch.tensor([-10.,0.,10.])
    tau=m.bounded_friction_torque(speed,.01,.05,1/240,torch.full_like(speed,1e-4))
    torch.testing.assert_close(tau,torch.tensor([.01,0.,-.01]))

def test_small_step_recovers_original_smooth_friction():
    speed=torch.linspace(-.2,.2,101,dtype=torch.float64)
    tau=m.bounded_friction_torque(speed,.01,.05,1/10000,torch.full_like(speed,1e-4))
    torch.testing.assert_close(tau,-.01*torch.tanh(speed/.05))

def test_invalid_time_step_rejected():
    with pytest.raises(ValueError):m.bounded_friction_torque(torch.ones(1),.01,.05,0,torch.ones(1))


def test_implicit_damping_reconstructs_law_and_has_finite_rest_limit():
    speed=torch.tensor([-10.,-.1,0.,.1,10.],dtype=torch.float64)
    damping=m.friction_damping(speed,.01,.05)
    assert torch.isfinite(damping).all() and (damping>=0).all()
    torch.testing.assert_close(damping[2],torch.tensor(.2,dtype=torch.float64))
    torch.testing.assert_close(-damping*speed,-.01*torch.tanh(speed/.05))


def test_implicit_damping_matches_constant_load_equilibrium():
    speed=torch.zeros(1,dtype=torch.float64)
    for _ in range(240):
        damping=m.friction_damping(speed,.01,.05)
        speed=(1.4e-4*speed+.002/240)/(1.4e-4+damping/240)
    expected=.05*torch.atanh(torch.tensor(.2,dtype=torch.float64))
    torch.testing.assert_close(speed[0],expected,atol=1e-9,rtol=1e-6)
