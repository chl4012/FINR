import numpy as np
from scipy.ndimage import label, generate_binary_structure
from scipy.fftpack import fftn, fftshift, ifftn
from scipy.ndimage import label
from scipy.stats import mode
import torch

def bwlabeln(A, conn=None):
    """
    Label connected components in binary image.
    
    Parameters:
    - A: Input binary image (numpy array). Can be of any dimension.
    - conn: Connectivity (optional). If not specified, the default is 8 for 2D, 26 for 3D,
            and maximal connectivity for higher dimensions.
            
    Returns:
    - L: Label matrix, with the same size as A.
    - num: Number of connected components found in A.
    """
    # Ensure the input is binary
    A = (A != 0)
    
    # Determine the default connectivity if not provided
    if conn is None:
        conn = max(3, A.ndim)
        conn = generate_binary_structure(A.ndim, conn)
    else:
        if isinstance(conn, int):
            conn = generate_binary_structure(A.ndim, conn)
        elif isinstance(conn, np.ndarray):
            if conn.shape != (3,) * A.ndim:
                raise ValueError("CONN must be a 3-by-3-by-...-by-3 matrix of 0s and 1s.")
    
    # Label the connected components
    L, num = label(A, structure=conn)
    
    return L, num



def sphere_kernel(matrix_size, voxel_size, radius):
    """
    Generate a sphere kernel in frequency space.
    
    Parameters:
    - matrix_size: A tuple indicating the size of the 3D matrix (e.g., (64, 64, 64)).
    - voxel_size: A tuple indicating the size of the voxel in each dimension.
    - radius: The radius of the sphere.
    
    Returns:
    - y: The FFT of the sphere kernel.
    """

    # Create meshgrid
    Y, X, Z = np.meshgrid(np.arange(-matrix_size[1]//2, matrix_size[1]//2),
                          np.arange(-matrix_size[0]//2, matrix_size[0]//2),
                          np.arange(-matrix_size[2]//2, matrix_size[2]//2))

    X = X * voxel_size[0]
    Y = Y * voxel_size[1]
    Z = Z * voxel_size[2]

    Sphere_out = ((np.maximum(np.abs(X) - 0.5 * voxel_size[0], 0) ** 2) +
                  (np.maximum(np.abs(Y) - 0.5 * voxel_size[1], 0) ** 2) +
                  (np.maximum(np.abs(Z) - 0.5 * voxel_size[2], 0) ** 2)) > radius ** 2

    Sphere_in = (((np.abs(X) + 0.5 * voxel_size[0]) ** 2) +
                 ((np.abs(Y) + 0.5 * voxel_size[1]) ** 2) +
                 ((np.abs(Z) + 0.5 * voxel_size[2]) ** 2)) <= radius ** 2

    Sphere_mid = np.zeros(matrix_size)

    # Precision control
    split = 10  # Error controlled at <1/(2*10)
    X_v, Y_v, Z_v = np.meshgrid(np.arange(-split + 0.5, split - 0.5 + 1),
                                np.arange(-split + 0.5, split - 0.5 + 1),
                                np.arange(-split + 0.5, split - 0.5 + 1))
    
    X_v = X_v / (2 * split)
    Y_v = Y_v / (2 * split)
    Z_v = Z_v / (2 * split)

    shell = 1 - Sphere_in - Sphere_out
    X = X[shell == 1]
    Y = Y[shell == 1]
    Z = Z[shell == 1]
    shell_val = np.zeros(X.shape)

    for i in range(len(X)):
        xx = X[i]
        yy = Y[i]
        zz = Z[i]

        occupied = ((xx + X_v * voxel_size[0]) ** 2 +
                    (yy + Y_v * voxel_size[1]) ** 2 +
                    (zz + Z_v * voxel_size[2]) ** 2) <= radius ** 2
        shell_val[i] = np.sum(occupied) / X_v.size

    Sphere_mid[shell == 1] = shell_val

    Sphere = Sphere_in + Sphere_mid
    Sphere = Sphere / np.sum(Sphere)

    y = fftn(fftshift(Sphere))
    return y



def SMV(iFreq, *args):
    """
    Apply the Spherical Mean Value (SMV) filter to the input frequency domain data.

    Parameters:
    - iFreq: Input 3D frequency data (numpy array).
    - *args: Variable arguments for either passing the kernel directly or generating it.

    Returns:
    - y: The filtered data.
    - K: The spherical kernel used for filtering.
    """

    if len(args) == 1:
        K = args[0]
    else:
        matrix_size = args[0]
        voxel_size = args[1]
        if len(args) < 3:
            radius = round(6 / max(voxel_size)) * max(voxel_size)  # default radius is 6mm
        else:
            radius = args[2]
        K = sphere_kernel(matrix_size, voxel_size, radius)

    y = ifftn(fftn(iFreq) * K)
    return y, K



def GenMask(iField, voxel_size, thresh):
    """
    Generate a binary mask for an image field with optional boundary erosion.
    Parameters:

    - iField: Input 4D complex-valued numpy array (e.g., (X, Y, Z, N) where N is the number of channels).
    - voxel_size: A tuple representing the voxel size in each dimension.
    - thresh: Threshold value for boundary erosion.

    Returns:
    - Mask: Binary mask after processing.
    """

    # Calculate the magnitude of the complex field
    #iMag = np.sqrt(np.sum(np.abs(iField) ** 2, axis=3))
    iMag = np.abs(iField)
    
    # Simple threshold to generate initial binary mask
    m = iMag > (0.015 * np.max(iMag))
    
    # Apply the SMV function to erode the boundary
    m1, _ = SMV(m, m.shape, voxel_size, 5) 
    m1 = m1 > thresh
    """
    # Label connected components and select the largest one
    l, num_labels = label(m1, structure=np.ones((3, 3, 3)))
    l_nonzero = l[l != 0]
    if len(l_nonzero) > 0:
       
        mode_value = mode(l_nonzero).mode[0]
        # Use the mode value to create the mask
        Mask = (l == mode_value)
    else:
        Mask = np.zeros_like(l)
    """
    Mask = m1
    # Optionally, restore the erosion (this step is commented out in the original MATLAB code)
    # Mask1 = SMV(Mask, m.shape, voxel_size, 10) > 0.001

    return Mask


def grad(chi, voxel_size=None):
    if voxel_size is None:
        voxel_size = [1, 1, 1]
    # Ensure chi is a double tensor
    chi = chi.double()
    # Calculate the gradient along the x-axis
    Dx = torch.cat((chi[1:, :, :], chi[-1:, :, :]), dim=0) - chi
    Dx = Dx / voxel_size[0]
    # Calculate the gradient along the y-axis
    Dy = torch.cat((chi[:, 1:, :], chi[:, -1:, :]), dim=1) - chi
    Dy = Dy / voxel_size[1]
    # Calculate the gradient along the z-axis
    Dz = torch.cat((chi[:, :, 1:], chi[:, :, -1:]), dim=2) - chi
    Dz = Dz / voxel_size[2]
    # Combine the gradients along the three axes
    Gx = torch.stack((Dx, Dy, Dz), dim=3)
    return Gx



def gradient_mask(iMag, Mask, voxel_size, percentage=0.9):
    
    field_noise_level = 0.01 * torch.max(iMag)
    wG = torch.abs(grad(iMag * (Mask > 0), voxel_size))
    denominator = torch.sum(Mask == 1).item()
    numerator = torch.sum(wG > field_noise_level).item()
    
    if (numerator / denominator) > percentage:
        while (numerator / denominator) > percentage:
            field_noise_level *= 1.05
            numerator = torch.sum(wG > field_noise_level).item()
    else:
        while (numerator / denominator) < percentage:
            field_noise_level *= 0.95
            numerator = torch.sum(wG > field_noise_level).item()
    
    wG = (wG <= field_noise_level)
    
    return wG

def erode_mask(error, Edge_mask, percentage=0.25):
    
    field_noise_level = 0.9 * torch.max(error*Edge_mask)
    wG = error*Edge_mask
    denominator = torch.sum(Edge_mask == 1).item()
    numerator = torch.sum(wG > field_noise_level).item()
    
    if (numerator / denominator) > percentage:
        while (numerator / denominator) > percentage:
            field_noise_level *= 1.05
            numerator = torch.sum(wG > field_noise_level).item()
    else:
        while (numerator / denominator) < percentage:
            field_noise_level *= 0.95
            numerator = torch.sum(wG > field_noise_level).item()
    
    wG = (wG >= field_noise_level)
    
    return wG