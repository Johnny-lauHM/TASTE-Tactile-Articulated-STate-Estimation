import unittest
import numpy as np
from rubik_lab.diagnostics.contact_wrench import gather_contacts,validate_normal_sum,ContactWrenchRecorder


class ContactWrenchTests(unittest.TestCase):
    def normal(self):
        force=np.array([[2.],[1.],[0.],[0.]])
        point=np.array([[0.,1.,0.],[0.,-1.,0.],[0.,0.,0.],[0.,0.,0.]])
        normal=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
        return [force,point,normal,np.zeros((4,1))],np.array([[2,0]]),np.array([[0,100]])

    def test_normal_buffer_mapping_and_sum(self):
        fields,count,start=self.normal();rows=gather_contacts(fields,count,start,0)
        self.assertEqual(rows.shape,(2,9));self.assertEqual(validate_normal_sum(rows,np.array([[1.,0.,0.],[0.,0.,0.]])),0.)
        with self.assertRaises(ValueError):validate_normal_sum(rows,np.array([[-1.,0.,0.],[0.,0.,0.]]))

    def test_moment_requires_individual_points(self):
        fields,count,start=self.normal();rows=gather_contacts(fields,count,start,0)
        forces=rows[:,1,None]*rows[:,5:8]
        moment=np.cross(rows[:,2:5],forces).sum(0)
        np.testing.assert_allclose(moment,[0,0,-3])
        self.assertFalse(np.allclose(moment,np.cross(rows[:,2:5].mean(0),forces.sum(0))))

    def test_invalid_range_and_saturation_rejected(self):
        fields,count,start=self.normal()
        with self.assertRaises(ValueError):gather_contacts(fields,np.array([[2,0]]),np.array([[3,0]]),0)
        with self.assertRaises(ValueError):gather_contacts(fields,np.array([[4,0]]),np.array([[0,0]]),0)

    def test_nonfinite_active_row_rejected(self):
        fields,count,start=self.normal();fields[1][0,0]=np.nan
        with self.assertRaises(ValueError):gather_contacts(fields,count,start,0)

    def test_friction_uses_own_indices_and_shapes(self):
        force=np.array([[99.,99.,99.],[1.,2.,3.],[0.,0.,0.],[0.,0.,0.]])
        point=np.zeros((4,3));rows=gather_contacts([force,point],np.array([[0,1]]),np.array([[0,1]]),0)
        np.testing.assert_allclose(rows[0,:4],[1,1,2,3])
        recorder=ContactWrenchRecorder(7);normal=np.empty((0,9));recorder.append(0,54,(normal,rows,np.zeros(16),np.zeros((54,3))))
        arrays=recorder.arrays();self.assertEqual(arrays['probe_normal_contacts'].shape,(0,12));self.assertEqual(arrays['probe_friction_contacts'].shape,(1,10));self.assertEqual(arrays['probe_wrench_frames'].shape,(1,19))

if __name__=='__main__':unittest.main()
