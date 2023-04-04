import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt


def calc_polygon_dist(x, y):
    dist = [0]
    for i in range(len(x) - 1):
        dist.append(np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2))
    dist = np.cumsum(dist)
    return dist


def points_to_polygon(x, y):
    points = [[x[i], y[i]] for i in range(len(x))]
    return Polygon(points)


class PolygonSampler:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.dist = calc_polygon_dist(x=x, y=y)
        self.interp_x = interp1d(x=self.dist, y=x)
        self.interp_y = interp1d(x=self.dist, y=y)

    def sample_dist(self, d, coords=False):
        x_sample = self.interp_x(d)
        y_sample = self.interp_y(d)

        if coords is True:
            return x_sample, y_sample
        else:
            return Point([x_sample, y_sample])

    def sample_n(self, n, coords=False):
        d = np.linspace(0, self.margin_length, n)

        x_sample = self.interp_x(d)
        y_sample = self.interp_y(d)

        if coords is True:
            return x_sample, y_sample
        else:
            return points_to_polygon(x=x_sample, y=y_sample)

    @property
    def margin_length(self):
        return self.dist[-1]

    @property
    def geometry(self):
        return points_to_polygon(x=self.x, y=self.y)

    @classmethod
    def from_polygon(cls, polygon):
        x, y = polygon.exterior.coords.xy
        x, y = list(x), list(y)
        return cls(x, y)


def dict_keys_to_int(dict):
    sort_idx = np.argsort(list(dict.keys()))
    values_sorted = np.array(list(dict.values()))[sort_idx]
    dict_int = {}
    for idx, value in enumerate(values_sorted):
        dict_int[idx] = value

    return dict_int


def calc_polygon_mapping(interior_polygon, exterior_polygon, one_to_one=True):
    interior_coords = np.array(interior_polygon.exterior.coords)
    exterior_coords = np.array(exterior_polygon.exterior.coords)
    sample_size = len(interior_coords)

    exterior_mapping = {i: exterior_coords[i] for i in range(len(exterior_coords))}

    ext_to_int_mapping = {
        i: np.argmin(
            np.linalg.norm(
                np.array(interior_coords) - np.array(exterior_coords[i]),
                axis=1,
            )
        )
        for i in range(sample_size)
    }
    if one_to_one is False:
        return ext_to_int_mapping

    else:
        int_idx_mapped = np.unique(np.sort(np.array(list(ext_to_int_mapping.values()))))

        int_idx_missed = np.diff(int_idx_mapped) - 1
        int_insert_idx_pairs = list(
            zip(
                int_idx_mapped[list(int_idx_missed > 0) + [False]],
                int_idx_mapped[[False] + list(int_idx_missed > 0)],
            )
        )
        int_insert_idx_counts = int_idx_missed[int_idx_missed > 0]

        if int_idx_mapped[0] != 0:
            pair = (int_idx_mapped[-1], int_idx_mapped[0])
            count = sample_size - int_idx_mapped[-1] - 1 + int_idx_mapped[0]

            int_insert_idx_pairs.append(pair)
            int_insert_idx_counts = np.append(
                int_insert_idx_counts,
                count,
            )

        ext_insert_idx_pairs = []
        # look through pairs, to work out which ext idx they each connect to
        for int_idx_min, int_idx_max in int_insert_idx_pairs:
            int_to_ext_min = []
            int_to_ext_max = []

            # find connecting pairs from original mapping
            # must also check for multiple connections, though
            # and we want to find the one closest to right link
            for ext_idx, int_idx in ext_to_int_mapping.items():
                if int_idx == int_idx_min:
                    int_to_ext_min.append(ext_idx)

            int_to_ext_min = np.array(int_to_ext_min)

            for item in int_to_ext_min:
                # this accounts for break at the idx origin
                if not ((item + 1) % sample_size in int_to_ext_min):
                    end_idx = item
            # one we have the left ext idx, we know the right ext idx
            ext_insert_idx_pairs.append([end_idx, (end_idx + 1) % sample_size])

        new_ext_coords = {}
        insert_mapping = {}
        for pair_idx in range(len(int_insert_idx_pairs)):
            num_new_ext_points = int_insert_idx_counts[pair_idx]
            int_idx_1, int_idx_2 = int_insert_idx_pairs[pair_idx]
            ext_idx_1, ext_idx_2 = ext_insert_idx_pairs[pair_idx]

            dn = 1 / (1 + num_new_ext_points)
            ext_x_1, ext_y_1 = exterior_coords[ext_idx_1]
            ext_x_2, ext_y_2 = exterior_coords[ext_idx_2]

            x_dist, y_dist = (ext_x_2 - ext_x_1) / (num_new_ext_points + 1), (
                ext_y_2 - ext_y_1
            ) / (num_new_ext_points + 1)
            for i in range(num_new_ext_points):
                ext_new_x, ext_new_y = ext_x_1 + x_dist * (i + 1), ext_y_1 + y_dist * (
                    i + 1
                )
                new_ext_idx = ext_idx_1 + dn * (1 + i)
                exterior_mapping[new_ext_idx] = (ext_new_x, ext_new_y)
                ext_to_int_mapping[new_ext_idx] = (int_idx_1 + 1 + i) % sample_size

        exterior_mapping = dict_keys_to_int(exterior_mapping)
        ext_to_int_mapping = dict_keys_to_int(ext_to_int_mapping)
        new_exterior_coords = np.asarray(list(exterior_mapping.values()))
        new_exterior_polygon = Polygon(new_exterior_coords)

        return ext_to_int_mapping, new_exterior_polygon


def calc_close_holes(poly):
    if poly.interiors:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly


def calc_angle(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    angle = np.arctan2(det, dot)
    return angle / (np.pi * 2)


def calc_power(sin_dist, nmax):
    n = np.zeros_like(sin_dist)
    if sin_dist.ndim == 0:
        if sin_dist >= 0:
            n = 1 + sin_dist * (nmax - 1)
        else:
            n = sin_dist * (1 - (1 / nmax)) + 1
    else:
        n[sin_dist >= 0] = 1 + sin_dist[sin_dist >= 0] * (nmax - 1)
        n[sin_dist < 0] = sin_dist[sin_dist < 0] * (1 - (1 / nmax)) + 1

    return n


class PolygonInterp:
    def __init__(
        self, interior_polygon, exterior_polygon, original_exterior_polygon, mapping
    ):
        self.interior_polygon = interior_polygon
        self.exterior_polygon = exterior_polygon
        self.original_exterior_polygon = original_exterior_polygon
        self.mapping = mapping

    @property
    def ext_coords(self):
        return np.array(self.exterior_polygon.exterior.coords)

    @property
    def int_coords(self):
        return np.array(self.interior_polygon.exterior.coords)

    @property
    def ext_centre_point(self):
        return np.array([coord[0] for coord in self.exterior_polygon.centroid.xy])

    def interp_angle(self, centre_point, angle):
        dist = np.sin(2 * np.pi * angle)

    def interp(
        self,
        dist,
        smoothing=None,
        no_holes=True,
        intersect_check=True,
        angle_async=False,
        async_power=1,
        async_angle_offset=None,
        centre_point=None,
    ):
        if angle_async is True:
            if centre_point is None:
                centre_point = self.ext_centre_point
            if async_angle_offset is None:
                async_angle_offset = 0

            v1 = np.array([0, 1])
            v2 = self.ext_coords - centre_point
            angle = calc_angle(v1=v1, v2=v2.T)
            async_value = np.sin(2 * np.pi * angle - async_angle_offset)
            async_power = calc_power(async_value, async_power)

            dist = (dist**async_power).reshape(-1, 1)

        ext_coords = self.ext_coords[list(self.mapping.keys())]
        int_coords = self.int_coords[list(self.mapping.values())]

        interp_coords = int_coords + ((ext_coords - int_coords) * dist)

        # post-processing
        interp_polygon = Polygon(interp_coords)
        interp_polygon = interp_polygon.buffer(0)

        if smoothing:
            interp_polygon = interp_polygon.buffer(smoothing, join_style=1).buffer(
                -smoothing, join_style=1
            )

        if intersect_check is True:
            interp_polygon = interp_polygon.union(self.interior_polygon)
            interp_polygon = interp_polygon.intersection(self.exterior_polygon)

        if interp_polygon.geom_type == "MultiPolygon":
            areas = np.asarray([item.area for item in list(interp_polygon.geoms)])
            interp_polygon = list(interp_polygon.geoms)[areas.argmax()]

        if no_holes is True:
            interp_polygon = calc_close_holes(interp_polygon)

        return interp_polygon

    def plot_angle(self, async_angle_offset=None, centre_point=None):
        if centre_point is None:
            centre_point = np.array(
                [coord[0] for coord in self.exterior_polygon.centroid.xy]
            )
        if async_angle_offset is None:
            async_angle_offset = 0

        v1 = centre_point + np.array([0, 1]) - centre_point
        v2 = self.ext_coords - centre_point
        angle = calc_angle(v1=v1, v2=v2.T)
        async_value = np.sin(2 * np.pi * angle + async_angle_offset)

        fig, ax = plt.subplots()
        ax.scatter(*self.int_coords.T)
        ax.scatter(*self.ext_coords.T, c=async_value, cmap="RdBu")
        return fig, ax

    @classmethod
    def from_polygons(cls, interior_polygon, exterior_polygon, sample_size):
        interior_sampler = PolygonSampler.from_polygon(interior_polygon)
        exterior_sampler = PolygonSampler.from_polygon(exterior_polygon)

        interior_polygon = interior_sampler.sample_n(n=sample_size)
        exterior_polygon = exterior_sampler.sample_n(n=sample_size)

        mapping, exterior_polygon_remapped = calc_polygon_mapping(
            interior_polygon=interior_polygon,
            exterior_polygon=exterior_polygon,
            one_to_one=True,
        )

        return cls(
            interior_polygon=interior_polygon,
            exterior_polygon=exterior_polygon_remapped,
            original_exterior_polygon=exterior_polygon,
            mapping=mapping,
        )
