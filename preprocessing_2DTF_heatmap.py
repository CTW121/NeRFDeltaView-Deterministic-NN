import vtk
import numpy as np
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

import os
import time
from datetime import datetime

from helpers import helpers


def similarity_vectors(vector1, vector2):
    """
    Calculate the angle in degrees between two vectors.
    """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cos_similarity = dot_product / (norm_vector1 * norm_vector2)

    # Calculate the angle in degrees
    angle = np.arccos(cos_similarity) * 180.0 / np.pi

    return angle, cos_similarity


def main():

    colors = vtk.vtkNamedColors()

    dataset = "chair"            # dataset: lego / hotdog / chair
    dataset_size = "full"       # full or partial
    iterations = 200000

    data_folder = "data"

    opacity_file_name = "{}_{}_{}_opacity.vtk".format(dataset, dataset_size, iterations)
    opacity_file_path = os.path.join(data_folder, opacity_file_name)
    opacity_volume, opacity_reader = helpers.vtk_read_volume_from_file(opacity_file_path)

    uncertainty_file_name = "{}_{}_{}_uncertainty.vtk".format(dataset, dataset_size, iterations)
    uncertainty_file_path = os.path.join(data_folder, uncertainty_file_name)
    uncertainty_volume, uncertainty_reader = helpers.vtk_read_volume_from_file(uncertainty_file_path)

    """
    Isosurface
    """
    isosurface_value = 0.90
    
    contour_opacity = helpers.vtk_contour_filter(opacity_volume, filter_value=isosurface_value)
    mapper_opacity = helpers.vtk_poly_data_mapper(contour_opacity)
    actor_opacity = helpers.vtk_create_actor(mapper_opacity, 'Green')

    """
    Uncertainty
    """
    # opacity_tf_uncertainty = helpers.vtk_create_piecewise_function([[0.00, 0.00], [0.85, 1.00]])
    opacity_tf_uncertainty = helpers.vtk_create_piecewise_function([[0.00, 0.00], [0.25, 1.00]])
    alpha_tf_uncertainty = helpers.vtk_create_color_transfer_function("RGB", [[0.00, 1.0, 1.0, 1.0], [1.00, 1.0, 0.0, 0.0]])
    volume_property_uncertainty = helpers.vtk_volume_property(alpha_tf_uncertainty, opacity_tf_uncertainty)
    volume_mapper_uncertainty = helpers.vtk_volume_ray_cast_mapper(uncertainty_reader)
    volume_uncertainty = helpers.vtk_create_volume(volume_mapper_uncertainty, volume_property_uncertainty)

    """
    Renderers
    """
    # isosurface
    interactor_isosurface = vtk.vtkRenderWindowInteractor()
    render_window_isosurface = vtk.vtkRenderWindow()
    renderer_isosurface = vtk.vtkRenderer()

    render_window_isosurface.AddRenderer(renderer_isosurface)
    interactor_isosurface.SetRenderWindow(render_window_isosurface)

    renderer_isosurface.AddVolume(actor_opacity)
    renderer_isosurface.SetBackground(colors.GetColor3d('White'))

    # uncertainty
    interactor_uncertainty = vtk.vtkRenderWindowInteractor()
    render_window_uncertainty = vtk.vtkRenderWindow()
    renderer_uncertainty = vtk.vtkRenderer()

    render_window_uncertainty.AddRenderer(renderer_uncertainty)
    interactor_uncertainty.SetRenderWindow(render_window_uncertainty)

    renderer_uncertainty.AddVolume(volume_uncertainty)
    renderer_uncertainty.SetBackground(colors.GetColor3d('Black'))


    camera = renderer_isosurface.GetActiveCamera()
    original_orient = helpers.vtk_get_orientation(renderer_isosurface)
    print("original_orient: ", original_orient["orientation"])


    azimuth = [i for i in range(0, 360+1, 15)]    # east-west
    elevation = [i for i in range(0, 360+1, 15)]  # north-south
    azimuth_len = len(azimuth)
    elevation_len = len(elevation)

    means_uncertainty = np.zeros((azimuth_len, elevation_len))
    standard_deviations_uncertainty = np.zeros((azimuth_len, elevation_len))

    z_buffer_data_isosurface = vtk.vtkFloatArray()
    z_buffer_data_uncertainty = vtk.vtkFloatArray()

    for i in range(azimuth_len):
        for j in range(elevation_len):
            helpers.vtk_set_orientation(renderer_isosurface, original_orient)
            helpers.vtk_set_orientation(renderer_uncertainty, original_orient)

            camera.Azimuth(azimuth[i]) # east-west
            camera.Elevation(elevation[j]) # north-south

            # https://discourse.vtk.org/t/vtkrenderer-error/6143/2
            view_up_vector = camera.GetViewUp()
            view_plane_normal = camera.GetViewPlaneNormal()
            # print("view_up_vector: ", view_up_vector)
            # print("view_plane_normal: ", view_plane_normal)
            # print("angle", angle_between_vectors(view_up_vector, view_plane_normal))

            angle, cos_similarity = similarity_vectors(view_up_vector, view_plane_normal)
            # print("azimuth: {}  | elevation: {} | angle: {:.2f} | cosine similarity: {:.2f}".format(azimuth[i], elevation[j], angle, cos_similarity))

            if abs(cos_similarity) > 0.95:
                camera.SetViewUp(0.0, 0.0, 1.0)
            else:
                camera.SetViewUp(0.0, 1.0, 0.0)
                
            renderer_isosurface.ResetCamera()
            renderer_uncertainty.SetActiveCamera(camera)
            renderer_uncertainty.ResetCamera()

            # === Z-buffer === #
            renderer_isosurface.PreserveDepthBufferOff()
            renderer_isosurface.GetRenderWindow().Render()

            renderer_uncertainty.PreserveDepthBufferOff()
            renderer_uncertainty.GetRenderWindow().Render()

            xmax_isosurface, ymax_isosurface = renderer_isosurface.GetRenderWindow().GetActualSize()
            renderer_isosurface.GetRenderWindow().GetZbufferData(0, 0, ymax_isosurface-1, xmax_isosurface-1, z_buffer_data_isosurface)

            xmax_uncertainty, ymax_uncertainty = renderer_uncertainty.GetRenderWindow().GetActualSize()
            renderer_uncertainty.PreserveDepthBufferOn()
            renderer_uncertainty.GetRenderWindow().GetZbufferData(0, 0, ymax_uncertainty-1, xmax_uncertainty-1, z_buffer_data_uncertainty)
            renderer_uncertainty.GetRenderWindow().SetZbufferData(0, 0, ymax_uncertainty-1, xmax_uncertainty-1, z_buffer_data_isosurface)

            # renderer_isosurface.GetRenderWindow().Render()
            # renderer_uncertainty.GetRenderWindow().Render()
            # ================ #

            render_window_isosurface.Render()
            render_window_uncertainty.Render()

            mean, standard_deviation = helpers.mean_standard_deviation(azimuth_len, elevation_len, render_window_uncertainty)

            means_uncertainty[i][j] = mean
            standard_deviations_uncertainty[i][j] = standard_deviation

    means_file = os.path.join(data_folder, "uncertainty_means.csv")
    stddev_file = os.path.join(data_folder, "uncertainty_standard_deviations.csv")

    np.savetxt(means_file, means_uncertainty, delimiter=",")
    np.savetxt(stddev_file, standard_deviations_uncertainty, delimiter=",")

if __name__ == "__main__":
    start_time = datetime.now()
    starting_time = start_time.strftime("%H:%M:%S")
    print("Start time: ", starting_time)

    main()

    end_time = datetime.now()
    ending_time = end_time.strftime("%H:%M:%S")
    print("End time: ", ending_time)

    print('Duration: {}'.format(end_time - start_time))