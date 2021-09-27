import navis
from tqdm import tqdm
from random import randrange
import cloudvolume
import time
from typing import Union, Tuple
import numpy as np
from itertools import chain


def random_sample_from_volume(
    volume: navis.Volume,
    supervoxels: cloudvolume.frontends.precomputed.CloudVolumePrecomputed,
    segmentation: cloudvolume.frontends.graphene.CloudVolumeGraphene,
    amount_to_query: Union[float, int] = 10,
    mip_level: Tuple[int, int, int] = (64, 64, 40),
    n_points: int = int(1e6),
    bbox_dim: int = 20,
    disable_rand_point_progress: bool = False
):

    """
    Takes a FAFB14 volume and generates random points within it.
    These points are then used to create a series of small bounding boxes.
    The supervoxels within these bounding boxes are then queried.
    These supervoxel ids are then mapped to root ids.

    Parameters
    ----------

    volume: a navis.Volume object retrieved either through FAFB14 or Janelia Hemibrain.
                If the latter, then you need to transform the volume into flywire
                space using a bridging transformation before using this function

    mip_level: the mip resolution level.
                Available resolutions are (16, 16, 40), (32, 32, 40), (64, 64, 40).
                I recommend the lowest voxel resolution (64, 64, 40).

    n_points: the number of randomly generated points to create.
                Note that not all of these will be in the volume, although those that are are kept.

    bbox_dim: dimension of the query cube - small values (<100) are encouraged.

    supervoxels: A supervoxel cloudvolume.Volume object
                specifying the supervoxel instance you are querying.

    e.g. cloudvolume.CloudVolume("precomputed://https://s3-hpcrc.rc.princeton.edu/fafbv14-ws/ws_190410_FAFB_v02_ws_size_threshold_200",
                                mip=mip_level)

    segmentation: A segmentation cloudvolume.Volume object specifying the segmentation instance you are querying.
                e.g. cloudvolume.CloudVolume("graphene://https://prodv1.flywire-daf.com/segmentation/table/fly_v31")

    amount_to_query: Either a float or an int.
                If a float, this is a percentage of the total number of randomly generated points that are in the volume you want to query
                If an int, the first n randomly generated points that are in the volume will be used to query.

    Returns
    ----------

    root_ids: an array of root ids that were within the randomly generated query boxes.


    Example
    ----------
    import pymaid
    import cloudvolume
    import numpy as np
    import navis
    from random import randrange
    import chain
    import time

    MB_R = pymaid.get_volume('MB_whole_R')
​
    mip_level = (64, 64, 40)
​
    cv_svs = cloudvolume.CloudVolume("precomputed://https://s3-hpcrc.rc.princeton.edu/fafbv14-ws/ws_190410_FAFB_v02_ws_size_threshold_200", mip=mip_level)

    cv_seg = cloudvolume.CloudVolume("graphene://https://prodv1.flywire-daf.com/segmentation/table/fly_v31")

    root_ids = random_sample_from_volume(volume = MB_R, mip_level = mip_level,
                                 n_points = int(1e6), bbox_dim = 20,
                                 supervoxels=cv_svs, segmentation=cv_seg)

    """
    start_time = time.time()

    assert isinstance(volume, navis.Volume), f'You need to pass a navis.Volume object. You passed {type(volume)}.'
    assert isinstance(supervoxels, cloudvolume.frontends.precomputed.CloudVolumePrecomputed), f'You need to pass a cloudvolume.frontends.precomputed.CloudVolumePrecomputed object. You passed {type(supervoxels)}.'
    assert isinstance(segmentation, cloudvolume.frontends.graphene.CloudVolumeGraphene), f'You need to pass a cloudvolume.frontends.graphene.CloudVolumeGraphene object. You passed {type(segmentation)}.'

    if type(n_points) != int:

        n_points = int(n_points)

    volume.vertices /= mip_level

    vertices_array = np.array(volume.vertices)
    vertices_array = vertices_array.astype(int)

    # generating random points
    n_point_string = format(n_points, ',d')
    print(f'Generating {n_point_string} random points... \n')

    # finding the min and max values of each xyz dimension
    x_min = min(vertices_array[:, 0])
    x_max = max(vertices_array[:, 0])

    y_min = min(vertices_array[:, 1])
    y_max = max(vertices_array[:, 1])

    z_min = min(vertices_array[:, 2])
    z_max = max(vertices_array[:, 2])

    # randomly generating integers inbetween these max and min values
    rand_x = [randrange(x_min, x_max) for i in tqdm(range(n_points), disable = disable_rand_point_progress)]
    rand_y = [randrange(y_min, y_max) for i in tqdm(range(n_points), disable = disable_rand_point_progress)]
    rand_z = [randrange(z_min, z_max) for i in tqdm(range(n_points), disable = disable_rand_point_progress)]

    xyz_arr = np.array([[rand_x, rand_y, rand_z]])
    xyz_arr = xyz_arr.T
    xyz_arr = xyz_arr.reshape(n_points, 3)

    # How many randomly generated points are in the volume?
    in_vol = navis.in_volume(xyz_arr, volume)
    print(f"""Of {n_point_string} random points generated, {xyz_arr[in_vol].shape[0] / n_points * 1e2 :.2f}% of them are in the volume.""")
    print(f"""This equals {xyz_arr[in_vol].shape[0]} points. \n""")
    xyz_in_vol = xyz_arr[in_vol]

    # generating query cubes
    xyz_start = xyz_in_vol
    xyz_end = xyz_in_vol + bbox_dim

    # querying flywire
    print('Querying flywire... \n')

    supervoxel_ids = []

    if isinstance(amount_to_query, float):

        assert amount_to_query <= 1.0, '''If using percentages of the total number of randomly generated points,
                                        you cannot use more than 100% of the points generated.
                                        Did you intend to use an integer?'''

        print(f'You are passing a float, so using {amount_to_query * 1e2}% of the total number ({len(xyz_start)}) of randomly points generated. \n')

        if amount_to_query == 1.0:

            print('You are using 100% of the randomly generated points - this can take a long time to complete.')

        n_query = int(len(xyz_start) * amount_to_query)

        print(f'{amount_to_query * 1e2}% = {n_query} points')

        xyz_start = xyz_start[:n_query]
        xyz_end = xyz_end[:n_query]

        print(f'Coverage: {((bbox_dim ** 3) * n_query / volume.volume) * 1e2 :.2g} % of the total {volume.name} volume is covered with {n_query} bounding boxes of {bbox_dim} cubic dimensions. \n')

        for i in range(n_query):

            q = supervoxels[xyz_start[i][0]: xyz_end[i][0],
                            xyz_start[i][1]: xyz_end[i][1],
                            xyz_start[i][2]: xyz_end[i][2]]

            supervoxel_ids.append(q)

    elif isinstance(amount_to_query, int):

        assert amount_to_query <= len(xyz_start), f'''You cannot use the first {amount_to_query} randomly generated
                                                    points when you have only {len(xyz_start)} exist in the volume.
                                                    Increase the number of randomly generated points in n_points.'''

        print(f'You are passing an integer, so the first {amount_to_query} randomly generated points will be used to query. \n')

        xyz_start = xyz_start[:amount_to_query]
        xyz_end = xyz_end[:amount_to_query]

        print(f'Coverage: {((bbox_dim ** 3) * amount_to_query / volume.volume) * 1e2 :.2g} % of the total {volume.name} volume is covered with {amount_to_query} bounding boxes of {bbox_dim} cubic dimensions. \n')

        for i in range(amount_to_query):

            q = supervoxels[xyz_start[i][0]: xyz_end[i][0],
                            xyz_start[i][1]: xyz_end[i][1],
                            xyz_start[i][2]: xyz_end[i][2]]

            supervoxel_ids.append(q)

    sv_ids_unique = [np.unique(i) for i in supervoxel_ids]
    sv_ids_unique = np.unique(list(chain.from_iterable(sv_ids_unique)))
    print('Fetching root ids... \n')
    root_ids = segmentation.get_roots(sv_ids_unique)

    # Removing zeros
    root_ids = root_ids[~(root_ids == 0)]

    root_ids = np.unique(root_ids)

    print(f'Random sampling complete: {len(root_ids)} unique root ids found \n')
    print(f'This function took {(time.time() - start_time) / 60 :.2f} minutes to complete')
    return(root_ids)
