import numpy as np
from scipy.ndimage import binary_erosion, median_filter, binary_fill_holes
from shapely.geometry import Polygon
import geopandas as gpd
from tqdm.auto import tqdm
from pathlib import Path

from icesea2.gridtools import poly_to_grid
from icesea2.iotools import check_output_dir


def margin_maker(margin_mask):
    margin_erosion = margin_mask - binary_erosion(margin_mask)
    margin_path_store = np.zeros_like(margin_erosion, dtype=int)
    uninspected_cells_x, uninspected_cells_y = np.where(
        (margin_erosion == 1) & (margin_path_store == 0)
    )
    paths = []
    while uninspected_cells_x.size != 0:
        initial_x, initial_y = uninspected_cells_x[0], uninspected_cells_y[0]
        path, margin_path_store = local_inspector(
            margin_erosion, initial_x, initial_y, margin_path_store
        )
        paths.append(path)
        uninspected_cells_x, uninspected_cells_y = np.where(
            (margin_erosion == 1) & (margin_path_store == 0)
        )
    return paths


def local_inspector(
    margin_erosion, centre_x, centre_y, margin_path_store, path_index=1, path=None
):
    if path is None:
        path = []

    if (path_index == 1) and (margin_erosion[centre_x, centre_y] == 1):
        margin_path_store[centre_x, centre_y] = path_index
        path.append([centre_x, centre_y, path_index])
        path_index = path_index + 1

    elif path_index == 1:
        raise ValueError

    local_inspection = margin_erosion[
        centre_x - 1 : centre_x + 2, centre_y - 1 : centre_y + 2
    ]
    check_coords = list(zip(*np.where(local_inspection > 0)))
    check_coords.remove((1, 1))
    for coord in check_coords:
        margin_x_index = centre_x + coord[0] - 1
        margin_y_index = centre_y + coord[1] - 1
        margin_store_value = margin_path_store[margin_x_index, margin_y_index]
        if margin_store_value != 0:
            pass
        else:
            margin_path_store[margin_x_index, margin_y_index] = path_index
            if path is None:
                path = []
            path.append([margin_x_index, margin_y_index, path_index])
            local_inspector(
                margin_erosion,
                margin_x_index,
                margin_y_index,
                margin_path_store,
                path_index + 1,
                path,
            )

    return path, margin_path_store


def path_to_polygon(paths, x_coords, y_coords, crs, to_latlon=True):
    polygon_geom = []
    for path in paths:
        path = np.asarray(path)
        path = path[path[:, 2].argsort()]

        # x_indices, y_indices = [entry[0] for entry in path], [entry[1] for entry in path]
        coords = [
            (x_coords[path[:, 1][index]], y_coords[path[:, 0][index]])
            for index in range(len(path))
        ]
        if not len(coords) < 3:
            polygon_geom.append(Polygon(coords))

    polygon = gpd.GeoDataFrame(
        index=list(range(len(polygon_geom))), crs=crs, geometry=polygon_geom
    )
    if to_latlon:
        polygon = polygon.to_crs("epsg:4326")

    return polygon


def smooth_raster(raster, smoothness=30):
    smooth_raster = median_filter(np.asarray(raster, dtype=float), size=smoothness)
    smooth_raster[smooth_raster > 0] = 1
    return smooth_raster


def generate_margins(
    margin_mask, margin_x, margin_y, raster_x, raster_y, raster_crs, smoothness=30
):

    margins_raw = []
    margins_smooth = []
    for index in tqdm(range(len(margin_mask))):
        # extract margin path from model
        margin_raw_paths = margin_maker(margin_mask[index])
        # convert to polygon in latlon
        margin_raw_polygon = path_to_polygon(
            paths=margin_raw_paths,
            x_coords=margin_x,
            y_coords=margin_y,
            crs="epsg:4326",
        )
        margins_raw.append(margin_raw_polygon)

        # rasterise onto laea grid
        margin_raster = poly_to_grid(
            polygons=margin_raw_polygon.to_crs(raster_crs),
            grid_x=raster_x,
            grid_y=raster_y,
        )
        # smooth raster
        smooth_margin_raster = smooth_raster(
            raster=margin_raster, smoothness=smoothness
        )
        # exract margin from smooth raster
        smooth_margin_paths = margin_maker(smooth_margin_raster)
        # convert to polygon in latlon
        smooth_margin_polygon = path_to_polygon(
            paths=smooth_margin_paths,
            x_coords=raster_x,
            y_coords=raster_y,
            crs=raster_crs,
        )
        margins_smooth.append(smooth_margin_polygon)

    return margins_raw, margins_smooth


def gdf_to_shp(gdf_objs, times, output_dir, write_times=False, overwrite=False):
    output_dir = check_output_dir(output_dir, overwrite)

    if isinstance(gdf_objs, (list, tuple)):
        assert len(times) == len(gdf_objs)
        write_times = True
    else:
        gdf_objs = [gdf_objs]
        times = [times]

    gdf_objs_nonempty = []
    times_nonempty = []
    for index, time in enumerate(times):
        if not gdf_objs[index].empty:
            gdf_objs_nonempty.append(gdf_objs[index])
            times_nonempty.append(time)
        else:
            print(f"Empty DataFrame: Skipping {time}")

    for index, time in enumerate(tqdm(times_nonempty)):
        output_dir_time = output_dir / f"{int(time)}"
        output_dir_time.mkdir(parents=True, exist_ok=True)
        gdf_objs_nonempty[index].to_file(str(output_dir_time / "margin.shp"))

    if write_times is True:
        with open(str(output_dir / "times"), "w") as f:
            for time in times_nonempty:
                f.write(f"{time}\n")


def shp_to_icesheet(
    shp_files,
    times,
    output_crs,
    output_dir,
    write_times=False,
    write_splits=False,
    overwrite=False,
):
    output_dir = check_output_dir(output_dir, overwrite)
    if isinstance(shp_files, (list, tuple)):
        assert len(times) == len(shp_files)
        write_times = True
    else:
        shp_files = [shp_files]
        times = [times]

    for index, shp_file in enumerate(shp_files):
        time = times[index]
        margin_gdf = gpd.read_file(shp_file).to_crs(output_crs)
        split_files = []
        for sub_margin_index, sub_margin in margin_gdf.iterrows():
            x, y = sub_margin["geometry"].exterior.coords.xy
            x = np.array(x)
            y = np.array(y)
            split_file = output_dir / f"{time}.{sub_margin_index+1}"
            with open(str(split_file), "w") as file_obj:
                for line_index in range(len(x)):
                    file_obj.write(f"{x[line_index]:.5f}\t{y[line_index]:.5f}\n")
            split_files.append(str(split_file.name))

        if write_splits is True:
            with open(output_dir / f"{time}.splits", "w") as file_obj:
                for path in split_files:
                    file_obj.write(f"{path}\n")

    if write_times is True:
        with open(str(output_dir / "times"), "w") as f:
            for time in times:
                f.write(f"{time}\n")


def shp_series_to_icesheet(shp_dir, icesheet_crs, output_dir, times=None):
    shp_dir = Path(shp_dir)
    if times is None:
        times = np.genfromtxt(str(shp_dir / "times"), dtype=int)

    shp_files = [str(shp_dir / str(time) / "margin.shp") for time in times]

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    shp_to_icesheet(
        shp_files=shp_files,
        times=times,
        output_crs=icesheet_crs,
        output_dir=output_dir,
        write_times=False,
        write_splits=False,
        overwrite=True,
    )


def get_local_array(array, x_idx, y_idx):
    local_array = array[y_idx - 1 : y_idx + 2, x_idx - 1 : x_idx + 2]
    return local_array.copy()


def calc_margin_array(ice_array, mask_region=None, min_thickness=None, all_touch=True):
    ice_mask = ice_array.copy()

    if mask_region is not None:
        ice_mask = ice_mask * mask_region

    if min_thickness is not None:
        ice_mask[ice_mask < min_thickness] = 0

    ice_mask[ice_mask > 0] = 1
    ice_mask = binary_fill_holes(ice_mask).astype(int)

    interior = binary_erosion(ice_mask)

    # erode interior to leave just margin cells
    exterior = ice_mask - interior
    # set margins to value of 2
    exterior = exterior * 2
    # set interior cells to value of 1
    margin_array = exterior + interior

    if all_touch is True:
        # search margin cells to make sure they all touch the interior
        y_idxs, x_idxs = np.where(margin_array == 2)
        filtered_idx = []
        for index in range(len(y_idxs)):
            y_idx, x_idx = y_idxs[index], x_idxs[index]
            local_area = get_local_array(margin_array, x_idx, y_idx)
            if 1 not in local_area:
                filtered_idx.append(index)

        margin_array[y_idxs[filtered_idx], x_idxs[filtered_idx]] = 0

    return margin_array


straights = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
diagonals = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
ea_proj = "+ellps=WGS84 +proj=laea +lon_0=0.0 +lat_0=90 +x_0=0.0 +y_0=0.0 +no_defs"


def calc_diagonal_weight(margin_array_local, diagonal):
    interior_straights = np.array((margin_array_local * straights) == 1, dtype=int)
    if np.array_equal(diagonal, [0, 0]):
        weight = interior_straights[0, 1] + interior_straights[1, 0]
    elif np.array_equal(diagonal, [0, 2]):
        weight = interior_straights[0, 1] + interior_straights[1, 2]
    elif np.array_equal(diagonal, [2, 0]):
        weight = interior_straights[1, 0] + interior_straights[2, 1]
    elif np.array_equal(diagonal, [2, 2]):
        weight = interior_straights[2, 1] + interior_straights[1, 2]

    if weight == 0:
        weight = 9
    return weight


def calc_margin_cell_order(margin_array):

    margin_arrays = []
    ordered_margin_arrays = []

    y_idxs, x_idxs = np.where(margin_array == 2)
    shape_coords = []
    while y_idxs.size > 0:
        ordered_margin_array = np.zeros_like(margin_array)

        # pick the first point to begin search
        cell_order = 1
        margin_point_y, margin_point_x = y_idxs[0], x_idxs[0]
        ordered_margin_array[margin_point_y, margin_point_x] = cell_order

        while True:
            new_point = None
            margin_array_local = get_local_array(
                margin_array, margin_point_x, margin_point_y
            )
            ordered_margin_array_local = get_local_array(
                ordered_margin_array, margin_point_x, margin_point_y
            )

            margins_array_local_mask = ordered_margin_array_local.copy()
            # mask the margins with 1
            margins_array_local_mask[ordered_margin_array_local > 0] = 1
            # where we've already picked margins, set to 0
            margin_array_local[margins_array_local_mask == 1] = 0
            # where we could have margins, return boolean
            possible_margins = margin_array_local == 2

            straights_idx = np.where((straights * possible_margins) == 1)
            # if we have a straight connection, pick the first one
            if straights_idx[0].size > 0:
                new_point = (straights_idx[0][0], straights_idx[1][0])
            else:
                diagonals_idx = np.where((diagonals * possible_margins) == 1)
                # if we have one diagonal connection, pick that one
                if diagonals_idx[0].size == 1:
                    new_point = (diagonals_idx[0][0], diagonals_idx[1][0])
                # if we have multiple, calculate preferred choice and pick that
                elif diagonals_idx[0].size > 1:
                    diagonals_idx = np.array(diagonals_idx).T
                    weights = []
                    for diagonal_idx in diagonals_idx:
                        weights.append(
                            calc_diagonal_weight(margin_array_local, diagonal_idx)
                        )

                    weights_idx = np.argmin(weights)
                    new_point = (
                        diagonals_idx[weights_idx][0],
                        diagonals_idx[weights_idx][1],
                    )

            if new_point:
                cell_order = cell_order + 1
                margin_point_y, margin_point_x = (
                    new_point[0] + margin_point_y - 1,
                    new_point[1] + margin_point_x - 1,
                )
                ordered_margin_array[margin_point_y, margin_point_x] = cell_order

            else:
                break

        margin_array[ordered_margin_array > 0] = 0
        margin_arrays.append(margin_array.copy())
        y_idxs, x_idxs = np.where(margin_array == 2)
        ordered_margin_arrays.append(ordered_margin_array)

    return ordered_margin_arrays


def calc_margin_polygon(
    margin_arrays, grid_x, grid_y, grid_crs, to_crs=None, min_area=None
):

    shapes = []
    for margin_array in margin_arrays:
        coords = []
        for index in range(int(np.max(margin_array))):
            y_idx, x_idx = np.where((margin_array == (index + 1)))
            x = grid_x[x_idx]
            y = grid_y[y_idx]
            coords.append((x, y))
        coords.append(coords[0])
        shapes.append(Polygon(coords))

    gdf = gpd.GeoDataFrame(geometry=shapes)
    gdf = gdf.set_crs(grid_crs)
    if to_crs:
        gdf = gdf.to_crs(to_crs)

    if min_area:
        gdf_ea = gdf.copy()
        gdf_ea = gdf_ea.to_crs(ea_proj)
        gdf = gdf[gdf_ea.area >= min_area]

    return gdf


def calc_topo_mask(topo_array, max_depth=-1000):
    topo_mask = topo_array.copy()
    topo_mask = np.where(topo_mask > max_depth, 1, 0)
    return topo_mask


def calc_margin(
    ice_array,
    grid_x,
    grid_y,
    grid_crs,
    smooth=False,
    mask_region=None,
    min_area=None,
    min_av_thickness=None,
    min_thickness=None,
):
    margin_array = calc_margin_array(
        ice_array, mask_region=mask_region, min_thickness=min_thickness
    )
    margin_arrays_ordered = calc_margin_cell_order(margin_array)
    margin_gdf = calc_margin_polygon(
        margin_arrays=margin_arrays_ordered,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_crs=grid_crs,
        min_area=min_area,
    )

    if min_av_thickness is not None:
        av_thickness_filter = []
        for index in range(len(margin_gdf)):
            mask = poly_to_grid(
                polygons=margin_gdf.iloc[index].geometry, grid_x=grid_x, grid_y=grid_y
            )
            if np.mean(np.ma.masked_where(mask == 0, ice_array)) < min_av_thickness:
                av_thickness_filter.append(False)
            else:
                av_thickness_filter.append(True)
        margin_gdf = margin_gdf[av_thickness_filter]

    if smooth is not False:
        smooth_grid_x = np.linspace(grid_x[0], grid_x[-1], 1000)
        smooth_grid_y = np.linspace(grid_y[0], grid_y[-1], 1000)
        margin_array_dense = poly_to_grid(
            polygons=margin_gdf, grid_x=smooth_grid_x, grid_y=smooth_grid_y
        )
        if smooth is True:
            size = (np.array(margin_array_dense.shape) / 30).round().astype(int)
        else:
            size = smooth

        margin_array_dense_smooth = median_filter(
            np.asarray(margin_array_dense, dtype=float), size=size
        )
        margin_gdf = calc_margin(
            ice_array=margin_array_dense_smooth,
            grid_x=smooth_grid_x,
            grid_y=smooth_grid_y,
            grid_crs=grid_crs,
            smooth=False,
        )

    return margin_gdf
