# A stripped-down version of HyperSpy and rosettasciio

The codes here are only used in docker image for `.mib` file conversion in
ePSIC. The dependencies are only `numpy`, `h5py` and `traits`. It includes
`BaseSignal`, `Signal2D` and `AxesManager` with only initialisation without any
interactivity and saving in `hspy` format.

The tests are limited to the minimal features and not comprehensive, and it is
tested against the official version of HyperSpy/rosettasciio.

**Do not use the codes unless you know what you are doing**, or you may find
yourself cursing at the FedID uef75971.
