import numpy as np
import torch
import torch.nn.functional as F 

def random_affine(x, min_rot=-30.0, max_rot=30.0, min_shear=-10.0,
                  max_shear=10.0, min_scale=0.8, max_scale=1.2):
   
    c, h, w = x.shape

    assert(len(x.shape) == 3)

    a = np.radians(np.random.rand() * (max_rot - min_rot) + min_rot)
    shear = np.radians(np.random.rand() * (max_shear - min_shear) + min_shear)
    scale = np.random.rand() * (max_scale -min_scale) + min_scale

    affine1_to_2 = np.array([[np.cos(a) * scale, -np.sin(a + shear) * scale, 0.],
                             [np.sin(a) * scale, np.cos(a + shear) * scale, 0.],
                             [0., 0., 1.]], dtype=np.float32)
    
    affine2_to_1 = np.linalg.inv(affine1_to_2).astype(np.float32)

    affine1_to_2, affine2_to_1 = affine1_to_2[:2, :], affine2_to_1[:2, :]
    affine1_to_2, affine2_to_1 = torch.from_numpy(affine1_to_2).cuda(), torch.from_numpy(affine2_to_1).cuda()

    x = perform_affine_tf(x.unsqueeze(dim=0), affine1_to_2.unsqueeze(dim=0))
    x = x.squeeze(dim=0)

    return x, affine1_to_2, affine2_to_1


def perform_affine_tf(x, tf_matrices):

    n_i, k, h, w = x.shape
    n_i2, r, c = tf_matrices.shape

    assert(n_i == n_i2)
    assert(r == 2 and c == 3)

    grid = F.affine_grid(tf_matrices, x.shape)

    data_tf = F.grid_sample(x, grid,
                            padding_mode='zeros')
    
    return data_tf
