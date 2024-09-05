from itertools import product
import json
import logging
from math import floor
import os
import shutil
import sys

import blosc
import h5py
# from hyperspy.signals import Signal2D
# from hyperspy.misc.array_tools import rebin
# from hyperspy.signal import BaseSignal
import numpy as np

# C extensions
# from mib_prop import mib_props
# from fast_binning import fast_bin

def main():

    args = json.loads(sys.argv[1])

    mib_path = args['mib_path']
    no_reshaping = args['no_reshaping']
    use_fly_back = args['use_fly_back']
    known_shape = args['known_shape']
    Scan_X = args['Scan_X']
    Scan_Y = args['Scan_Y']
    iBF = args['iBF']
    bin_sig_flag = args['bin_sig_flag']
    bin_sig_factor = args['bin_sig_factor']
    bin_nav_flag = args['bin_nav_flag']
    bin_nav_factor = args['bin_nav_factor']
    create_json = args['create_json']
    ptycho_config = args['ptycho_config']
    ptycho_template = args['ptycho_template']

    print("Packages imported successfully!")


if __name__ == "__main__":
    main()
