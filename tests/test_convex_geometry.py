import unittest
import numpy as np
import trimesh
from rubik_lab.diagnostics.convex_geometry import convex_features,convex_overlap

class ConvexGeometryTest(unittest.TestCase):
    def box(self,size=1.):
        mesh=trimesh.creation.box(extents=[size]*3)
        return convex_features(mesh.vertices,mesh.faces)
    def overlap(self,a,b,shift):
        return convex_overlap(a,b,np.eye(3),np.zeros(3),np.eye(3),np.array(shift))
    def test_overlap_and_separation(self):
        self.assertAlmostEqual(self.overlap(self.box(),self.box(),[.9,0,0]),.1)
        self.assertAlmostEqual(self.overlap(self.box(),self.box(),[2,0,0]),-1.)
    def test_containment(self):
        self.assertAlmostEqual(self.overlap(self.box(),self.box(.1),[0,0,0]),.55)
    def test_rigid_frame_invariance(self):
        rotation=trimesh.transformations.rotation_matrix(.731,[1,2,3])[:3,:3]
        position=np.array([4.,-2.,1.])
        result=convex_overlap(self.box(),self.box(),rotation,position,rotation,position+rotation@np.array([.9,0,0]))
        self.assertAlmostEqual(result,.1)
    def test_rotated_cube(self):
        rotation=trimesh.transformations.rotation_matrix(np.pi/4,[0,0,1])[:3,:3]
        result=convex_overlap(self.box(),self.box(),np.eye(3),np.zeros(3),rotation,np.array([1.1,0,0]))
        self.assertAlmostEqual(result,.5+np.sqrt(.5)-1.1)

class TorchConvexBoxTest(unittest.TestCase):
    def test_matches_general_sat(self):
        import torch
        from rubik_lab.diagnostics.convex_geometry import convex_box_overlap_torch
        rng=np.random.default_rng(2401)
        mesh=trimesh.convex.convex_hull(rng.normal(size=(30,3))*.03)
        features=convex_features(mesh.vertices,mesh.faces)
        box=trimesh.creation.box(extents=[.07,.06,.04]);bf=convex_features(box.vertices,box.faces)
        tf=tuple(torch.tensor(v,dtype=torch.float64) for v in features)
        for i in range(30):
            r=trimesh.transformations.rotation_matrix(rng.uniform(-3,3),rng.normal(size=3))[:3,:3]
            br=trimesh.transformations.rotation_matrix(rng.uniform(-3,3),rng.normal(size=3))[:3,:3]
            pos=rng.uniform(-.08,.08,3);center=rng.uniform(-.02,.02,3)
            expected=convex_overlap(features,bf,r,pos,br,center)
            tensor=lambda x:torch.tensor(x,dtype=torch.float64)
            actual=convex_box_overlap_torch(tf,tensor(r),tensor(pos),tensor(br),tensor(center),tensor([.035,.03,.02]))
            self.assertAlmostEqual(float(actual),expected,places=10)
    def test_translation_gradient(self):
        import torch
        from rubik_lab.diagnostics.convex_geometry import convex_box_overlap_torch
        mesh=trimesh.creation.box();features=tuple(torch.tensor(v,dtype=torch.float64) for v in convex_features(mesh.vertices,mesh.faces))
        pos=torch.tensor([.9,0.,0.],dtype=torch.float64,requires_grad=True)
        eye=torch.eye(3,dtype=torch.float64)
        value=convex_box_overlap_torch(features,eye,pos,eye,torch.zeros(3,dtype=torch.float64),torch.full((3,),.5,dtype=torch.float64))
        value.backward()
        self.assertAlmostEqual(float(value),.1)
        np.testing.assert_allclose(pos.grad.numpy(),[-1,0,0],atol=1e-10)

class TorchConvexPairTest(unittest.TestCase):
    def test_analytic_overlap_separation_and_containment(self):
        import torch
        from rubik_lab.diagnostics.convex_geometry import convex_overlap_torch
        t=lambda x:torch.tensor(x,dtype=torch.float64)
        def box(size):
            m=trimesh.creation.box(extents=[size]*3)
            return tuple(t(v) for v in convex_features(m.vertices,m.faces))
        eye=t(np.eye(3));zero=t(np.zeros(3))
        for size,shift,expected in [(1.,.9,.1),(1.,2.,-1.),(.1,0.,.55)]:
            actual=convex_overlap_torch(box(1.),box(size),eye,zero,eye,t([shift,0.,0.]))
            self.assertAlmostEqual(float(actual),expected,places=10)

    def test_random_hulls_match_independent_numpy(self):
        import torch
        from rubik_lab.diagnostics.convex_geometry import convex_overlap_torch
        rng=np.random.default_rng(2402);features=[]
        for n in (20,25):
            m=trimesh.convex.convex_hull(rng.normal(size=(n,3))*.03)
            features.append(convex_features(m.vertices,m.faces))
        t=lambda x:torch.tensor(x,dtype=torch.float64)
        tensors=[tuple(t(v) for v in f) for f in features]
        for _ in range(20):
            rotations=[trimesh.transformations.rotation_matrix(rng.uniform(-3,3),rng.normal(size=3))[:3,:3] for _ in range(2)]
            positions=[rng.uniform(-.05,.05,3) for _ in range(2)]
            expected=convex_overlap(*features,rotations[0],positions[0],rotations[1],positions[1])
            actual=convex_overlap_torch(*tensors,t(rotations[0]),t(positions[0]),t(rotations[1]),t(positions[1]))
            self.assertAlmostEqual(float(actual),expected,places=10)

    def test_translation_gradient(self):
        import torch
        from rubik_lab.diagnostics.convex_geometry import convex_overlap_torch
        m=trimesh.creation.box();t=lambda x:torch.tensor(x,dtype=torch.float64)
        f=tuple(t(v) for v in convex_features(m.vertices,m.faces));eye=t(np.eye(3));zero=t(np.zeros(3))
        position=torch.tensor([.9,0.,0.],dtype=torch.float64,requires_grad=True)
        value=convex_overlap_torch(f,f,eye,position,eye,zero);value.backward()
        np.testing.assert_allclose(position.grad.numpy(),[-1,0,0],atol=1e-10)

    def test_rotation_and_translation_gradcheck(self):
        import torch
        from rubik_lab.hand_kinematics import axis_angle_to_mat
        from rubik_lab.diagnostics.convex_geometry import convex_overlap_torch
        rng=np.random.default_rng(2403);features=[]
        for n in (18,23):
            mesh=trimesh.convex.convex_hull(rng.normal(size=(n,3))*.02)
            features.append(tuple(torch.tensor(v,dtype=torch.float64) for v in convex_features(mesh.vertices,mesh.faces)))
        def value(x):
            rot=axis_angle_to_mat(x[3:][None])[0]
            return convex_overlap_torch(*features,rot,x[:3],torch.eye(3,dtype=x.dtype),x.new_tensor([.015,-.02,.01]))
        x=torch.tensor([.04,.01,-.005,.13,-.17,.21],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(value,(x,),eps=1e-7,atol=1e-6,rtol=1e-4))

if __name__=='__main__':unittest.main()
