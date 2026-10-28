"""Read-only selected-environment contact manifolds for diagnostic scoring.

Normal contact and friction buffers have independent counts/start indices.
Copy normal rows before requesting friction: the installed PhysX view reuses
its count/start buffers. Never infer a moment from the mean contact position.
"""
import numpy as np


def gather_contacts(fields,counts,starts,sensor_index):
    """Return [filter_index, concatenated point fields] for one sensor."""
    arrays=[np.asarray(x).reshape(len(x),-1) for x in fields]
    counts=np.asarray(counts);starts=np.asarray(starts)
    if counts.shape!=starts.shape or counts.ndim!=2:raise ValueError('Invalid contact buffer layout')
    capacity=len(arrays[0])
    if any(len(x)!=capacity for x in arrays):raise ValueError('Mismatched contact buffers')
    if int(counts.sum())>=capacity:raise ValueError('Contact buffer capacity saturated; completeness unknown')
    rows=[];used=[]
    for link,(count,start) in enumerate(zip(counts[sensor_index],starts[sensor_index])):
        count,start=int(count),int(start)
        if not count:continue
        if count<0 or start<0 or start+count>capacity:raise ValueError('Invalid contact buffer range')
        used.extend(range(start,start+count))
        row=np.column_stack([np.full(count,link),*(x[start:start+count] for x in arrays)])
        if not np.isfinite(row).all():raise ValueError('Nonfinite active contact data')
        rows.append(row)
    if len(used)!=len(set(used)):raise ValueError('Overlapping filtered contact ranges')
    width=1+sum(x.shape[1] for x in arrays)
    return np.concatenate(rows) if rows else np.empty((0,width))


def validate_normal_sum(rows,matrix):
    """Check sign, impulse-to-force scale and pair mapping against PhysX sum."""
    total=np.zeros_like(matrix,dtype=float)
    if len(rows):np.add.at(total,rows[:,0].astype(int),rows[:,1,None]*rows[:,5:8])
    error=float(np.linalg.norm(total-matrix,axis=-1).max())
    limit=max(.001,float(np.linalg.norm(matrix,axis=-1).max())*1e-4)
    if not np.isfinite(error) or error>limit:raise ValueError(f'Normal manifold/matrix mismatch {error} N > {limit} N')
    return error


class ContactWrenchRecorder:
    def __init__(self,env):
        self.env=int(env);self.normal=[];self.friction=[];self.frames=[];self.matrices=[]

    def capture(self,e):
        from rubik_lab.hand_kinematics import quat_to_mat
        cpu=lambda value:value.detach().cpu().numpy().copy()
        sensor=e.cube_contacts.sensors['top'];data=sensor.data;view=sensor.contact_physx_view
        matrix=cpu(data.force_matrix_w[self.env,0]);net=cpu(data.net_forces_w[self.env,0])
        if matrix.shape!=(54,3) or view.sensor_count!=e.num_envs:raise ValueError('Expected one top sensor per env and54 explicit hand filters')
        force,point,normal,separation,count,start=view.get_contact_data(dt=e.physics_dt)
        normal_rows=gather_contacts([cpu(x) for x in (force,point,normal,separation)],cpu(count),cpu(start),self.env)
        error=validate_normal_sum(normal_rows,matrix)
        force,point,count,start=view.get_friction_data(dt=e.physics_dt)
        friction_rows=gather_contacts([cpu(force),cpu(point)],cpu(count),cpu(start),self.env)
        cube=e.cube.data;i=self.env;j=e.twist_id
        position=cpu(cube.body_pos_w[i,e.core_id]);axis=cpu(quat_to_mat(cube.body_quat_w[i,e.core_id])[:,2])
        metadata=np.r_[position,axis,float(cube.joint_pos[i,j]),float(cube.joint_vel[i,j]),float(cube.joint_damping[i,j]),float(cube.joint_effort_target[i,j]),error,len(normal_rows),len(friction_rows),net]
        if not np.isfinite(metadata).all():raise ValueError('Nonfinite wrench frame state')
        return normal_rows,friction_rows,metadata,matrix

    def append(self,trial,step,captured):
        normal,friction,metadata,matrix=captured;prefix=np.array([trial,step,self.env],dtype=float)
        self.normal.append(np.column_stack([np.tile(prefix,(len(normal),1)),normal]))
        self.friction.append(np.column_stack([np.tile(prefix,(len(friction),1)),friction]))
        self.frames.append(np.r_[prefix,metadata]);self.matrices.append(matrix)

    def arrays(self):
        return dict(probe_normal_contacts=np.concatenate(self.normal) if self.normal else np.empty((0,12)),probe_friction_contacts=np.concatenate(self.friction) if self.friction else np.empty((0,10)),probe_wrench_frames=np.asarray(self.frames).reshape(-1,19),probe_normal_force_matrix=np.asarray(self.matrices).reshape(-1,54,3))

    @staticmethod
    def schema():
        return dict(normal_columns='trial,step,env,filter(0..26left/27..53right),normal_force_N,point_world_xyz,normal_world_xyz,separation_m',friction_columns='trial,step,env,filter,friction_force_world_xyz,point_world_xyz',frame_columns='trial,step,env,axis_origin_world_xyz,axis_world_xyz,layer_angle_rad,layer_speed_rad_s,joint_damping,explicit_joint_effort,normal_sum_error_N,normal_point_count,friction_point_count,unfiltered_top_normal_force_world_xyz',scope='Selectedenv topbody raw percontact data at completed40Hz controlframes; dt=physics_dt converts impulses toforces. No meanpoint moment approximation; no fullsubstep impulse or actorinput claim.')
