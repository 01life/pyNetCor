from ._core import *

__all__ = ["Canopy"]


class Canopy:
    """
    Canopy clustering.

    Parameters
    ----------
    t2 : float, optional, default 0.2
        Max tight distance (correlation difference). If the distance from a point to the canopyis is less
        than t2, the point is closed enough to the canopy center and is removed from the dataset.
    t1 : float, optional, default 0.6
        Max loose distance (correlation difference). If the distance from a point to the canopy is less
        than t1, the point is considered to be in the canopy.
    max_merge_distance : float, optional, default 0.2
        The maximum distance (correlation difference) between two canopy centers for merging the canopies
        should be noted. It is important to mention that the final canopy profiles are calculated after
        the merge step, and as a result, some final canopies may have profiles that are closer than the
        specified `max_merge_distance`.
    stop_criteria : int, optional, default 50000
        The clustering process will terminate after processing a specified number of seeds according to
        the stop criteria. Setting it to 0 will disable this particular stopping criterion.
    distance_measure : {'pearson', 'spearman'}, default 'pearson'
        The specified distance measure is utilized for clustering.
    random_seed : int, optional, default None
        The random seed is utilized to shuffle the data prior to clustering.
    threads : int, optional, default 8
        The number of threads to use.
    """

    def __init__(
        self,
        t1: float = 0.2,
        t2: float = 0.6,
        max_merge_distance: float = 0.2,
        stop_criteria: int = 50000,
        distance_measure: str = "pearson",
        random_seed: int = None,
        threads: int = 8,
    ):
        self._c = None
        self.t1 = t1
        self.t2 = t2
        self.max_merge_distance = max_merge_distance
        self.stop_criteria = stop_criteria
        self.distance_measure = distance_measure
        self.random_seed = random_seed
        self.threads = threads
        # The values are hard coded to avoid user confusion.
        self._min_step_distance = 0.001
        self._max_canopy_walk_num = 6

        if self.t1 < 0 or self.t1 > 2:
            raise ValueError("t1 must be between 0 and 2")
        if self.t2 < 0 or self.t2 > 2:
            raise ValueError("t2 must be between 0 and 2")
        if self.max_merge_distance < 0 or self.max_merge_distance > 2:
            raise ValueError("max_merge_distance must be between 0 and 2")
        if self.stop_criteria < 0:
            raise ValueError("stop_criteria must be greater than 0")
        if self.threads < 1:
            raise ValueError("threads must be greater than 0")

    def fit(self, x):
        """
        Compute canopy clustering.

        Parameters
        ----------
            x : array_like
                A 2-D array. Training data to cluster.
        """
        self._c = CanopyCluster(
            x,
            self.t1,
            self.t2,
            self.max_merge_distance,
            self._min_step_distance,
            self._max_canopy_walk_num,
            self.stop_criteria,
            self.distance_measure,
            self.random_seed,
            self.threads,
        )

    def fit_predict(self, x):
        """
        Computing the centroids of clusters and predicting the cluster assignment for each sample.

        Parameters
        ----------
            x : array_like
                A 2-D array. Training data to cluster.

        Returns
        -------
            ndarray
                Cluster assignment for each sample.
        """
        self.fit(x)
        return self._c.get_labels()

    def predict(self, x):
        """
        Predict the cluster that each sample in `x` is most likely to belong to.

        Parameters
        ----------
            x : array_like
                A 2-D array. New data to predict.

        Returns
        -------
            ndarray
                Cluster assignment for each sample.
        """
        return self._c.predict(x)

    @property
    def labels_(self):
        """
        Label of each point. Each point may belongs to a set of canopies (soft clustering).
        List[List[int]]
        """
        return self._c.get_labels()

    @property
    def best_labels_(self):
        """
        Label of each point. Each point may belongs to a set of canopies, the best label is
        is determined by minimizing the distance to the cluster center.
        ndarray
        """
        return self._c.get_best_labels()

    @property
    def cluster_centers_(self):
        """
        The cluster centers.
        ndarray.
        """
        return self._c.get_cluster_centers()
