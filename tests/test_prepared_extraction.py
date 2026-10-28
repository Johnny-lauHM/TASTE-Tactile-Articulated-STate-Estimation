import sys,json,tempfile,subprocess,unittest
from pathlib import Path
import h5py,numpy as np

class PreparedExtractionTests(unittest.TestCase):
    def test_only_initialization_rows_are_removed_and_protocol_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.h5';output=Path(folder)/'prepared.h5'
            source.with_suffix('.json').write_text(json.dumps({'sequence':None,'initial_settle_steps':2}))
            observations=np.arange(15,dtype=np.float32).reshape(5,3)
            with h5py.File(source,'w') as f:
                f['step']=np.arange(5);f['episode_uid']=np.zeros(5,dtype=np.int64)
                f['policy_obs']=observations;f['primitive_done']=[False,False,False,False,True]
            before=source.read_bytes()
            script=Path(__file__).resolve().parents[1]/'scripts/extract_prepared_primitive.py'
            subprocess.run([sys.executable,str(script),str(source),'--out',str(output)],check=True,capture_output=True)
            self.assertEqual(source.read_bytes(),before)
            with h5py.File(output,'r') as f:
                np.testing.assert_array_equal(f['step'][:],[0,1,2])
                np.testing.assert_array_equal(f['policy_obs'][:],observations[2:])
                self.assertEqual(json.loads(f.attrs['execution_protocol_json'])['initial_settle_steps'],2)
                self.assertEqual(f['primitive_done'][:].tolist(),[False,False,True])

if __name__=='__main__':unittest.main()
