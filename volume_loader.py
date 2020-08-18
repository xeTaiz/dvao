import os
import pydicom
import numpy as np
import dicom_numpy

from utils import hidden_errors
from tf_utils import *
from pathlib import Path

def read_dicom_folder(dicom_folder, rescale=None):
    ''' Reads all .dcm files in `dicom_folder` and merges them to one volume

    Returns:
        The volume and the affine transformation from pixel indices to xyz coordinates
    '''
    dss = [pydicom.dcmread(str(dicom_folder/dcm)) for dcm in os.listdir(dicom_folder) if dcm.endswith('.dcm')]
    vol, mat = dicom_numpy.combine_slices(dss, rescale)
    return vol, dss[0]

def get_largest_dir(dirs, minsize=100):
    ''' Returns the dir with the most files from `dirs`'''
    m = max(dirs, key=lambda d: len(os.listdir(d)) if os.path.isdir(d) else 0)
    if len(os.listdir(m)) >= minsize: return m
    else: return None

def get_volume_dirs(path):
    path = Path(path)
    return list(
        filter(lambda p: p is not None,
        map(   get_largest_dir,                             # extract subdir with most files in it (highest res volume)
        map(   lambda p: list(p.iterdir()),                 # get list of actual volume directorie
        map(   lambda p: next(p.iterdir())/'Unknown Study', # cd into subfolders CQ500-CT-XX/Unknown Study/
        filter(lambda p: p.is_dir(),                        # Get all dirs, no files
        path.iterdir())))))                                 # Iterate over path directory
)

def get_volume_gen(volume_dirs, rescale=None, tf_pts=None):
    ''' Make a generator that loads volumes from a list of volume directories, `volume_dirs`.
    Returns: (volume:np.ndarray , index_to_pos_4x4:np.ndarray) '''
    def vol_gen():
        for vol_dir in volume_dirs:
            with hidden_errors():
                try:
                    vol, dcm = read_dicom_folder(vol_dir, rescale)
                    vox_scl = np.array([dicom.PixelSpacing[0], dicom.PixelSpacing[1], dicom.SliceThickness]).astype(np.float32)
                    vox_scl /= vox_scl.min()
                    vol_name = str(vol_dir.parent.parent.parent.name)
                    if tf_pts is None:
                        peaks  = get_histogram_peaks(normalized_vol)
                        tf_pts = get_trapezoid_tf_points_from_peaks(peaks)
                except dicom_numpy.DicomImportException:
                    print(f'Could not load {vol_dir}')
                    continue
            yield vol, tf_pts, vox_scl, vol_name
    return vol_gen()

__all__ = ['read_dicom_folder', 'get_largest_dir', 'get_volume_gen', 'get_volume_dirs']
