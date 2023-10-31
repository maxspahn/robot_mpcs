import numpy as np
def quaternion_to_rotation_matrix(q):
    q /= np.linalg.norm(q)  # Normalize quaternion
    x, y, z, w = q
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x**2 + z**2), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x**2 + y**2)]
    ])
    return rotation_matrix