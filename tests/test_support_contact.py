import unittest
from types import SimpleNamespace
import numpy as np
import torch
from rubik_lab.diagnostics.support_contact import SupportContact
from rubik_lab.diagnostics.convex_geometry import convex_features


class SupportContactTests(unittest.TestCase):
    def setUp(self):
        import trimesh
        box=trimesh.creation.box(extents=[.002]*3)
        features=tuple(torch.tensor(v,dtype=torch.float64) for v in convex_features(box.vertices,box.faces))
        self.fk=SimpleNamespace(pad_links={f:f for f in ('thumb','middle')})
        self.features={f:features for f in ('thumb','middle')}
        self.frames={f:(torch.tensor([[sign*.0355,0.,-.01]],dtype=torch.float64),torch.eye(3,dtype=torch.float64)[None]) for f,sign in [('thumb',-1),('middle',1)]}
        self.pads={f:(self.frames[f][0],torch.tensor([[-sign,0.,0.]],dtype=torch.float64)) for f,sign in [('thumb',-1),('middle',1)]}
        self.model=SupportContact(self.fk,self.features,self.frames,self.pads)

    def moved(self,delta):
        frames={f:(p+torch.tensor([delta],dtype=p.dtype),r) for f,(p,r) in self.frames.items()}
        pads={f:(frames[f][0],n) for f,(_,n) in self.pads.items()}
        return frames,pads

    def test_initial_and_tangent_slide(self):
        values,_=self.model.measure(self.frames,self.pads)
        self.assertTrue(bool((values>=0).all()))
        frames,pads=self.moved([0.,.008,0.])
        values,details=self.model.measure(frames,pads)
        self.assertTrue(bool((values>=0).all()))
        self.assertAlmostEqual(float(details['thumb']['overlap_drift']),0.,places=12)
        self.assertGreater(float(torch.linalg.vector_norm(pads['thumb'][0]-self.pads['thumb'][0])),.004)

    def test_normal_departure_rejected(self):
        frames,pads=self.moved([.0001,0.,0.])
        values,_=self.model.measure(frames,pads)
        self.assertTrue(bool((values<0).all()))

    def test_face_edge_rejected(self):
        frames,pads=self.moved([0.,.034,0.])
        values,_=self.model.measure(frames,pads)
        self.assertTrue(bool((values<0).all()))

    def test_normal_change_rejected(self):
        pads={f:(p,-n) for f,(p,n) in self.pads.items()}
        values,_=self.model.measure(self.frames,pads)
        self.assertTrue(bool((values<0).all()))

    def test_opposite_initial_orientation_rejected(self):
        pads={f:(p,-n) for f,(p,n) in self.pads.items()}
        with self.assertRaises(ValueError):SupportContact(self.fk,self.features,self.frames,pads)

    def test_reserve_tightens_optimizer_only(self):
        model=SupportContact(self.fk,self.features,self.frames,self.pads,optimizer_reserve_m=5e-6)
        frames,pads=self.moved([47e-6,0.,0.])
        accepted,_=model.measure(frames,pads)
        optimized,_=model.measure(frames,pads,optimization=True)
        self.assertTrue(bool((accepted>=0).all()))
        self.assertTrue(bool((optimized<0).all()))
        with self.assertRaises(ValueError):SupportContact(self.fk,self.features,self.frames,self.pads,optimizer_reserve_m=50e-6)

    def test_contact_band_gradient(self):
        def values(x):
            shift=torch.stack([x,x*0,x*0])[None]
            frames={f:(p+shift,r) for f,(p,r) in self.frames.items()}
            pads={f:(frames[f][0],n) for f,(_,n) in self.pads.items()}
            return self.model.measure(frames,pads)[0]
        x=torch.tensor(.00002,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(values,(x,),eps=1e-8,atol=1e-5,rtol=1e-4))

if __name__=='__main__':unittest.main()
