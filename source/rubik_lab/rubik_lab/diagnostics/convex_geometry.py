"""Separating-axis checks for closed convex polyhedra, independent of FCL depths."""
import numpy as np


def _directions(vectors):
    vectors=np.asarray(vectors,dtype=np.float64)
    lengths=np.linalg.norm(vectors,axis=1)
    vectors=vectors[lengths>1e-12]/lengths[lengths>1e-12,None]
    if not len(vectors):return vectors.reshape(0,3)
    dominant=np.argmax(np.abs(vectors),axis=1)
    vectors*=np.where(vectors[np.arange(len(vectors)),dominant]<0,-1.,1.)[:,None]
    _,indices=np.unique(np.round(vectors,10),axis=0,return_index=True)
    return vectors[indices]


def convex_features(vertices,faces):
    """Prepare face normals and edge directions; input must describe a convex hull."""
    vertices=np.asarray(vertices,dtype=np.float64);faces=np.asarray(faces,dtype=int)
    if not np.isfinite(vertices).all():raise ValueError('Nonfinite vertices')
    triangles=vertices[faces]
    normals=_directions(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]))
    edges=_directions(np.concatenate([triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,1],triangles[:,0]-triangles[:,2]]))
    if not len(normals) or not len(edges):raise ValueError('Degenerate convex geometry')
    return vertices,normals,edges


def convex_overlap(first,second,first_rotation,first_position,second_rotation,second_position):
    """Signed SAT overlap in meters; negative means separated, not Euclidean distance.

    Positive value is minimum translation to separate the authored convex hulls.
    Does not include collision cooking, compliance, or contact/rest offsets.
    """
    av,an,ae=first;bv,bn,be=second
    av=av@np.asarray(first_rotation).T+first_position
    bv=bv@np.asarray(second_rotation).T+second_position
    an=an@np.asarray(first_rotation).T;ae=ae@np.asarray(first_rotation).T
    bn=bn@np.asarray(second_rotation).T;be=be@np.asarray(second_rotation).T
    axes=_directions(np.concatenate([an,bn,np.cross(ae[:,None,:],be[None,:,:]).reshape(-1,3)]))
    minimum=np.inf
    for start in range(0,len(axes),256):
        directions=axes[start:start+256].T
        a=av@directions;b=bv@directions
        overlaps=np.minimum(a.max(0)-b.min(0),b.max(0)-a.min(0))
        minimum=min(minimum,float(overlaps.min()))
    return minimum


def convex_box_overlap_torch(features,rotation,position,box_rotation,box_center,half):
    """Differentiable SAT overlap for one convex hull and oriented box.

    Features are local vertices/normals/edges as torch tensors. Positive means
    overlap; negative is a separating-axis gap, not Euclidean distance.
    """
    import torch
    vertices,normals,edges=features
    points=(vertices@rotation.T+position-box_center)@box_rotation
    normals=normals@rotation.T@box_rotation
    edges=edges@rotation.T@box_rotation
    basis=torch.eye(3,dtype=vertices.dtype,device=vertices.device)
    cross=torch.linalg.cross(edges[:,None,:],basis[None,:,:],dim=-1).reshape(-1,3)
    lengths=torch.linalg.vector_norm(cross,dim=-1)
    cross=cross[lengths>1e-12]/lengths[lengths>1e-12,None]
    axes=torch.cat([basis,normals,cross])
    projections=points@axes.T
    radius=(axes.abs()*half).sum(-1)
    return torch.minimum(projections.amax(0)+radius,radius-projections.amin(0)).amin()


def convex_overlap_torch(first,second,first_rotation,first_position,second_rotation,second_position):
    """Differentiable signed SAT overlap of two cooked convex hulls.

    Negative values are separating-axis gaps, not Euclidean distances. This
    function performs no FCL filtering and retains every nondegenerate axis.
    """
    import torch
    av,an,ae=first;bv,bn,be=second
    av=av@first_rotation.T+first_position
    bv=bv@second_rotation.T+second_position
    an=an@first_rotation.T;ae=ae@first_rotation.T
    bn=bn@second_rotation.T;be=be@second_rotation.T
    cross=torch.linalg.cross(ae[:,None,:],be[None,:,:],dim=-1).reshape(-1,3)
    lengths=torch.linalg.vector_norm(cross,dim=-1)
    cross=cross[lengths>1e-12]/lengths[lengths>1e-12,None]
    axes=torch.cat([an,bn,cross])
    a=av@axes.T;b=bv@axes.T
    return torch.minimum(a.amax(0)-b.amin(0),b.amax(0)-a.amin(0)).amin()
