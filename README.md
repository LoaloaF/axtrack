# AxTrack

![](./examples/example_timelapse.gif)

## Installation
First clone the repo and install the submodules:

    git clone --recursive https://github.com/LoaloaF/axtrack.git
    cd axtrack/libmot && python setup.py install develop --user
    cd ../pyastar2d && python setup.py install develop --user

Download model & example data:
              
    curl https://polybox.ethz.ch/index.php/s/Run8gaUVT45hmjX/download > deployed_model/E1000.pth
    curl https://polybox.ethz.ch/index.php/s/qlOKY3asRdcnmKh/download > examples/example_timelapse.tif

## Quickstart
Run the included example:\
`python example/test.py`

![](./examples/axon_dets/example_timelapse_dets.mp4)
