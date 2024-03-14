#!/usr/bin/env python

"""
Create a ImageData grid and then assign an uncertainty scalar value and opacity scalar value to each point.
"""

import argparse
import sys
import os
import time
from datetime import datetime

import vtk

import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkImagingCore import vtkImageCast
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkMath, vtkFloatArray
from vtkmodules.vtkIOLegacy import (
    vtkStructuredPointsReader,
    vtkStructuredPointsWriter,
    vtkRectilinearGridWriter,
)
import vtk.util.numpy_support as numpy_support
from vtkmodules.vtkRenderingVolume import (
    vtkFixedPointVolumeRayCastMapper,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper, vtkOpenGLGPUVolumeRayCastMapper
from vtkmodules.vtkCommonTransforms import (
    vtkTransform,
)
from vtkmodules.vtkFiltersGeneral import (
    vtkTransformFilter,
    vtkTransformPolyDataFilter,
)
from vtkmodules.vtkCommonDataModel import (
    vtkPolyData,
    vtkPiecewiseFunction,
    vtkRectilinearGrid,
    vtkImageData,
    vtkStructuredPoints,
    vtkPlane,
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkColorTransferFunction,
    vtkVolumeProperty,
    vtkVolume,
)
from vtkmodules.vtkFiltersGeometry import (
    vtkImageDataGeometryFilter,
    vtkRectilinearGridGeometryFilter,
)
from vtkmodules.vtkIOXML import (
    vtkXMLImageDataReader,
    vtkXMLImageDataWriter,
)
from vtkmodules.vtkIOLegacy import (
    vtkUnstructuredGridWriter,
    #vtkImageDataWriter,
)
from vtkmodules.vtkFiltersCore import (
    vtkContourFilter,
    vtkCutter,
    vtkPolyDataNormals,
    vtkStripper,
    vtkStructuredGridOutlineFilter,
    vtkTubeFilter,
    vtkClipPolyData,
)

import numpy as np
import torch
import random
import math
import yaml

from nerf import (CfgNode, models, helpers)

def main():

    ### *** user defined parameters for vtkImageData *** ###

    scene = 'mic'
    dataset = 'full'    # dataset size: 'full' or 'partial'/'selected' view dataset
    iteration = 200000

    xyzMin = -1.3
    xyzMax = 1.3
    xyzNumPoint = 100
    radiance_field_noise_std = 0.2
    
    ### *** user defined parameters for vtkImageData *** ###

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    
    # clear memory in GPU CUDA
    torch.cuda.empty_cache()

    model_fine = models.FlexibleNeRFModel(
        cfg.models.fine.num_layers,
        cfg.models.fine.hidden_size,
        cfg.models.fine.skip_connect_every,
        cfg.models.fine.num_encoding_fn_xyz,
        cfg.models.fine.num_encoding_fn_dir,
    )


    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
    else:
        sys.exit("Please enter the path of the checkpoint file.")
    
    model_fine.eval()


    include_input_xyz = 3 if cfg.models.fine.include_input_xyz else 0
    include_input_dir = 3 if cfg.models.fine.include_input_dir else 0
    dim_xyz = include_input_xyz + 2 * 3 * cfg.models.fine.num_encoding_fn_xyz
    dim_dir = include_input_dir + 2 * 3 * cfg.models.fine.num_encoding_fn_dir
    if not cfg.models.fine.use_viewdirs:
        dim_dir = 0

    print("Scene: ", scene)
    print("Dataset: ", dataset)
    print("Iteration: ", iteration)
    print("Points per dimension: ", xyzNumPoint)

    dxyz = abs(xyzMax - xyzMin) / (xyzNumPoint - 1)

    voxelVol = vtkStructuredPoints()
    #voxelVol = vtkImageData()
    voxelVol.SetDimensions(xyzNumPoint, xyzNumPoint, xyzNumPoint)
    voxelVol.SetOrigin(-(abs(xyzMax-xyzMin)/2), -(abs(xyzMax-xyzMin)/2), -(abs(xyzMax-xyzMin)/2))
    voxelVol.SetSpacing(dxyz, dxyz, dxyz)

    arrayUncertainty = vtkDoubleArray()
    arrayUncertainty.SetNumberOfComponents(1) # this is 3 for a vector
    arrayUncertainty.SetNumberOfTuples(voxelVol.GetNumberOfPoints())

    arrayDensity = vtkDoubleArray()
    arrayDensity.SetNumberOfComponents(1) # this is 3 for a vector
    arrayDensity.SetNumberOfTuples(voxelVol.GetNumberOfPoints())

    tensor_input = torch.zeros(1, dim_xyz+dim_dir)
    npoints = voxelVol.GetNumberOfPoints()

    # file_uncertainty = open('output_point_{}_uncertainty.txt'.format(iteration), 'w')
    # file_density = open('output_point_{}_density.txt'.format(iteration), 'w')

    for i in range(npoints):
        if i%100000 == 0:
            print("i: ", i)
        x, y, z = voxelVol.GetPoint(i)
        xyz  = [x, y, z]
        xyz_tensor = torch.Tensor([xyz])

        encode_pos = helpers.embedding_encoding(xyz_tensor, num_encoding_functions=cfg.models.fine.num_encoding_fn_xyz)
        tensor_input[..., : dim_xyz] = encode_pos

        if cfg.models.fine.use_viewdirs:
            encode_dir = helpers.embedding_encoding(xyz_tensor, num_encoding_functions=cfg.models.fine.num_encoding_fn_dir)
            # tensor_input[..., dim_xyz :] = encode_dir
            encode_dir_zeros = torch.zeros_like(encode_dir)
            tensor_input[..., dim_xyz :] = encode_dir_zeros

        output = model_fine(tensor_input)
        [output_list] = output.tolist()

        arrayUncertainty.SetValue(i, output_list[4])
        arrayDensity.SetValue(i, output_list[3])

        # file_uncertainty.write(str(output_list[4]) + '\n')
        # file_density.write(str(output_list[3]) + '\n')
    
    # file_uncertainty.close()
    # file_density.close()
    
    if radiance_field_noise_std > 0.0:
        noise = np.random.randn(npoints)
    else:
        noise = np.zeros_like(npoints)
    arraySigma_tensor = torch.nn.functional.relu(torch.Tensor(arrayDensity + noise))
    alpha_tensor = 1.0 - torch.exp(-arraySigma_tensor)
    alpha = numpy_support.numpy_to_vtk(num_array=alpha_tensor.numpy(), deep=True)

    voxelVolOpacity = vtkStructuredPoints()
    voxelVolOpacity.DeepCopy(voxelVol)
    voxelVolOpacity.GetPointData().SetScalars(alpha)

    writerOpacity = vtkStructuredPointsWriter()
    writerOpacity.WriteExtentOn()
    writerOpacity.SetFileName("{}_{}_{}_opacity.vtk".format(scene, dataset, iteration))
    writerOpacity.SetInputData(voxelVolOpacity)
    writerOpacity.Write()

    voxelVolUncertainty = vtkStructuredPoints()
    voxelVolUncertainty.DeepCopy(voxelVol)
    voxelVolUncertainty.GetPointData().SetScalars(arrayUncertainty)

    writerUncertainty = vtkStructuredPointsWriter()
    writerUncertainty.WriteExtentOn()
    writerUncertainty.SetFileName("{}_{}_{}_uncertainty.vtk".format(scene, dataset, iteration))
    writerUncertainty.SetInputData(voxelVolUncertainty)
    writerUncertainty.Write()

if __name__ == "__main__":
    start_time = datetime.now()
    starting_time = start_time.strftime("%H:%M:%S")
    print("Start time: ", starting_time)

    main()
    
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))