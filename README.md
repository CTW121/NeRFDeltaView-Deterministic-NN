# NeRFDeltaView Deterministic Neural Network

We developed a visualization tool for visualizing the uncertainty estimated by the [NeRF Uncertainty Deterministic Neural Network](https://github.com/CTW121/NeRF-Uncertainty-Deterministic-NN).

Uncertainty visualization provides users with an in-depth understanding of the data for analysis and to perform confident and informed decision-making. The main purpose of our tool is to highlight the significance of interactive visualization in enabling users to explore the estimated uncertainty in synthesized scenes, identify model limitations, and aid in understanding NeRF model uncertainty.

## Prerequisites

Ensure you have met the following requirements:
- You have a **Linux/Windows/Mac** machine.
- You have installed **Python 3.8 or higher**.
- You have installed [**PyTorch**](https://pytorch.org/).
- You have installed [**PyQt6**](https://doc.qt.io/qtforpython-6/).
- You have installed [**VTK from Python wrappers**](https://docs.vtk.org/en/latest/getting_started/index.html).

We recommend running the application using [**conda**](https://docs.conda.io/en/latest/).

## How to launch the Visualization Tool Application?

To launch the NeRFDeltaView Deterministic Neural Network (NN) visualization tool application, follow these steps:

1. Copy the trained model (checkpoint and yml files) from [NeRF Uncertainty Deterministic NN](https://github.com/CTW121/NeRF-Uncertainty-Deterministic-NN) to the [VTK_writer](https://github.com/CTW121/NeRFDeltaView-Deterministic-NN/tree/master/VTK_writer) folder.

2. Within the [VTK_writer](https://github.com/CTW121/NeRFDeltaView-Deterministic-NN/tree/master/VTK_writer) folder, execute `python vtk_writer.py` to generate the VTK 3D volumetric data files (estimated opacity and uncertainty). Then, copy these VTK 3D volumetric data files to the [data](https://github.com/CTW121/NeRFDeltaView-Deterministic-NN/tree/master/data) folder.

3. Run `python preprocessing_2DTF_heatmap.py` to produce CSV files containing color and density means, as well as standard deviations for heatmap visualization.

4. Run `python NeRFDeltaView.py` to launch the visualization tool application.

Following are the screenshots of the NeRFDeltaView Deterministic NN visualization tool application:
![NeRFDeltaView_Deterministic_NN_A](https://github.com/CTW121/NeRFDeltaView-Deterministic-NN/blob/master/images/NeRFDeltaView__Uncertainty_Neural_Network_A.png)

![NeRFDeltaView_Deterministic_NN_B](https://github.com/CTW121/NeRFDeltaView-Deterministic-NN/blob/master/images/NeRFDeltaView__Uncertainty_Neural_Network_B.png)

The demo videos can be found in [demo_video](https://github.com/CTW121/NeRFDeltaView-Ensemble/tree/master/demo_video). (Note: the demo videos are made from [NeRFDeltaView Ensemble](https://github.com/CTW121/NeRFDeltaView-Ensemble))

## Uncertainty Estimation
The uncertainty prediction is based on a single forward pass within a deterministic network. The deterministic network, represented as a multilayer perceptron (MLP) estimating color $\boldsymbol{c}$, volume density $\sigma$, and an uncertainty $\delta$. We adopt the uncertainty estimation method from Recursive Neural Radiance ([Recursive-NeRF](https://ieeexplore.ieee.org/document/9909994)). Refer to [NeRF Uncertainty Deterministic NN](https://github.com/CTW121/NeRF-Uncertainty-Deterministic-NN) and read [master's thesis]([https://github.com/CTW121/NeRFDeltaView-Deterministic-NN/blob/master/Report.pdf](https://github.com/CTW121/NeRFDeltaView-Deterministic-NN/blob/master/Uncertainty_Visualization_for_Neural_Radiance_Field__Chua_T.W.pdf)) for more detail.
