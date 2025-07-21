import torch

def get_3d_locations(d,h,w,device_):
    locations_x = torch.linspace(0, w-1, w).view(1, 1, 1, w).to(device_).expand(1, d, h, w)
    locations_y = torch.linspace(0, h-1, h).view(1, 1, h, 1).to(device_).expand(1, d, h, w)
    locations_z = torch.linspace(0, d-1,d).view(1, d, 1, 1).to(device_).expand(1, d, h, w)
    locations_x = (2.0*locations_x - (w-1))/(w-1)
    locations_y = (2.0*locations_y - (h-1))/(h-1)
    locations_z = (2.0*locations_z - (d-1))/(d-1)
    # stack locations
    locations_3d = torch.stack([locations_x, locations_y, locations_z], dim=4).view(-1, 3, 1)
    return locations_3d


def fftn(input):
    fft_x = torch.fft.fft(input, dim=0)
    fft_xy = torch.fft.fft(fft_x, dim=1)
    fft_xyz = torch.fft.fft(fft_xy, dim=2)
    return fft_xyz

def ifftn(input):
    ifft_x = torch.fft.ifft(input, dim=0)
    ifft_xy = torch.fft.ifft(ifft_x, dim=1)
    ifft_xyz = torch.fft.ifft(ifft_xy, dim=2)
    return ifft_xyz