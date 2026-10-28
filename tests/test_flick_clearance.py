import unittest
import torch
from rubik_lab.flick import rigid_finger_box_penetration
from rubik_lab.hand_kinematics import axis_angle_to_mat

class Points:
    def __init__(self,name,point):self.link_pts={name:torch.tensor([[point]],dtype=torch.float32)}
    def points(self,frames,links=None):return self.link_pts

class ClearanceTests(unittest.TestCase):
    def test_pad_top_compression_allowed_but_core_not(self):
        R=torch.eye(3)[None]
        self.assertEqual(float(rigid_finger_box_penetration(Points('left_index_elastomer',[0,0,.025]),{},'index',R)),0)
        self.assertGreater(float(rigid_finger_box_penetration(Points('left_index_elastomer',[0,0,0]),{},'index',R)),.01)
    def test_rigid_top_and_rotated_box(self):
        point=Points('right_index_DP',[.045,0,.025]);R=torch.eye(3)[None]
        self.assertEqual(float(rigid_finger_box_penetration(point,{},'index',R)),0)
        R=axis_angle_to_mat(torch.tensor([[0.,0.,torch.pi/4]]))
        self.assertGreater(float(rigid_finger_box_penetration(point,{},'index',R)),.003)
    def test_collision_depth_has_joint_optimization_gradient(self):
        point=Points('left_index_DP',[.034,0,.025]);point.link_pts['left_index_DP'].requires_grad_(True)
        depth=rigid_finger_box_penetration(point,{},'index',torch.eye(3)[None]);depth.sum().backward()
        self.assertLess(float(point.link_pts['left_index_DP'].grad[0,0,0]),0)

class SurfacePatchTests(unittest.TestCase):
    def test_rotated_patch_bounds_and_clockwise_torque(self):
        from rubik_lab.flick import surface_patch_target,H
        n=torch.tensor([0.,-1.,0.]);axis=torch.tensor([0.,0.,1.]);t=torch.tensor([1.,0.,0.])
        R=axis_angle_to_mat(torch.tensor([[0.,0.,.7],[0.,0.,-.3]]))
        q=torch.tensor([[1.,1.,1.],[-1.,-1.,-1.]])
        target=surface_patch_target(q,R,n,axis,t,-1.,(.019,.028),.001)
        local=torch.einsum('bji,bj->bi',R,target)
        self.assertTrue(((local[:,0]>=-.028001)&(local[:,0]<=-.018999)).all())
        torch.testing.assert_close(local[:,1],torch.full((2,),-(H-.001)))
        self.assertTrue(((local[:,2]>=H/3+.002999)&(local[:,2]<=H-.002999)).all())
        force=-torch.einsum('bij,j->bi',R,n)
        self.assertTrue((torch.linalg.cross(target,force)[:,2]<0).all())
    def test_surface_loss_keeps_normal_gradient_without_pin_to_material_point(self):
        from rubik_lab.flick import surface_patch_target
        q=torch.tensor([[-.023,-.040,.024]],requires_grad=True)
        target=surface_patch_target(q,torch.eye(3)[None],torch.tensor([0.,-1.,0.]),torch.tensor([0.,0.,1.]),torch.tensor([1.,0.,0.]),-1.,(.019,.028),.001)
        ((q-target)**2).sum().backward()
        self.assertLess(float(q.grad[0,1]),0)
        self.assertEqual(float(q.grad[0,0]),0.)
        self.assertEqual(float(q.grad[0,2]),0.)

if __name__=='__main__':unittest.main()
