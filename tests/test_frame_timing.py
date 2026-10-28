import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rubik_lab'))
from rubik_lab.frame_timing import SensorFrameClock

class FrameTimingTests(unittest.TestCase):
    def test_delay_and_sample_hold_age(self):
        clock=SensorFrameClock(2,'cpu',1/240)
        self.assertEqual(clock.observe(0).tolist(),[[0.,0.],[0.,0.]])
        clock.capture(0)
        self.assertFalse(clock.observe(4)[:,1].any())
        clock.deliver()
        self.assertTrue(torch.allclose(clock.observe(6),torch.tensor([[.025,1.],[.025,1.]])))
        self.assertTrue(torch.allclose(clock.observe(12)[:,0],torch.full((2,),.05)))

    def test_reset_pending_zero_frame_stays_invalid_until_new_capture(self):
        clock=SensorFrameClock(2,'cpu',1/240)
        clock.capture(8)
        clock.reset(torch.tensor([0]))
        clock.deliver()
        self.assertTrue(torch.allclose(clock.observe(14),torch.tensor([[0.,0.],[.025,1.]])))
        clock.capture(16);clock.deliver()
        self.assertTrue((clock.observe(22)[:,1]==1).all())

if __name__=='__main__':unittest.main()
