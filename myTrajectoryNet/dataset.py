""" dataset.py

Loads datasets into uniform format for learning continuous flows

"""
import re
import math
import numpy as np
import torch
import scipy.sparse


from sklearn.preprocessing import StandardScaler


class SCData(object):
    """ Base Class for single cell like trajectory data """

    def __init__(self):
        super().__init__()
        self.val_labels = []

    def has_validation_samples(self):
        return False

    def load(self):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def get_data(self, labels=None):
        raise NotImplementedError

    def get_ncells(self):
        raise NotImplementedError

    def get_velocity(self):
        raise NotImplementedError

    def has_velocity(self):
        raise NotImplementedError

    def leaveout_timepoint(self, tp):
        raise NotImplementedError

    def num_timepoints(self):
        raise NotImplementedError

    def known_base_density(self):
        """ Returns if the dataset starts from a known base density.

        Generally single cell datasets do not have a known base density
        where generated datasets do.
        """
        raise NotImplementedError

    def base_density(self): # 正規分布
        def standard_normal_logprob(z):
            logZ = -0.5 * math.log(2 * math.pi)
            return torch.sum(logZ - z.pow(2) / 2, 1, keepdim=True)

        return standard_normal_logprob

    def base_sample(self):
        return torch.randn

    def plot_density(self):
        import matplotlib.pyplot as plt

        npts = 100
        side = np.linspace(-4, 4, npts)
        xx, yy = np.meshgrid(side, side)
        xx = torch.from_numpy(xx).type(torch.float32)
        yy = torch.from_numpy(yy).type(torch.float32)
        z_grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
        logp_grid = data.base_density()(z_grid)
        plt.pcolormesh(xx, yy, np.exp(logp_grid.numpy()).reshape(npts, npts))
        plt.show()

    def plot_data(self):
        import scprep
        import matplotlib.pyplot as plt

        nbase = 5000
        all_data = np.concatenate(
            [data.get_data(), data.base_sample()(nbase, self.get_shape()[0]).numpy()],
            axis=0,
        )
        lbs = np.concatenate([data.get_times(), np.repeat(["Base"], nbase)])
        if all_data.shape[1] == 2:
            scprep.plot.scatter2d(all_data, c=lbs)
        else:
            fig, axes = plt.subplots(2, all_data.shape[1] // 2)
            axes = axes.flatten()
            for i in range(all_data.shape[1] - 1):
                scprep.plot.scatter2d(
                    all_data[:, i : i + 2],
                    c=lbs,
                    ax=axes[i],
                    xlabel="PC %d" % (i + 1),
                    ylabel="PC %d" % (i + 2),
                )
        plt.show()

    def plot_velocity(self):
        import matplotlib.pyplot as plt

        idx = np.random.randint(self.get_ncells(), size=200)
        data = self.get_data()[idx]
        velocity = self.velocity[idx]
        plt.quiver(data[:, 0], data[:, 1], velocity[:, 0], velocity[:, 1])
        plt.show()

    def plot_paths(self):
        paths = self.get_paths()
        paths = paths[:1000]
        import matplotlib.pyplot as plt

        for path in paths:
            plt.plot(path[:, 0], path[:, 1])
        plt.show()

    def factory(cfg):
        # Generated Circle datasets
        name = cfg['name']
        max_dim = cfg['max_dim']
        val_size = cfg['val_size']
        if name == "CIRCLE":
            return CircleTestData()
        if name == "CIRCLE2":
            return CircleTestDataV2()
        if name == "CIRCLE3":
            return CircleTestDataV3()
        if name == "CIRCLE4":
            return CircleTestDataV4()
        if name == "CIRCLE5":
            return CircleTestDataV5()
        if name == "TREE":
            return TreeTestData()
        if name == "CYCLE":
            return CycleDataset()

        # Generated sklearn datasets
        if name == "MOONS":
            return SklearnData("moons")
        if name == "SCURVE":
            return SklearnData("scurve")
        if name == "BLOBS":
            return SklearnData("blobs")
        if name == "CIRCLES":
            return SklearnData("circles")

        if name == "CLARK-PCA":
            return ClarkData("pcs", max_dim=max_dim)
        if name == "CLARK-PHATE":
            return ClarkData("phate")
        if name == "CLARK-UMAP":
            return ClarkData("original_embedding", max_dim=max_dim)

        if name == "EB":
            return EBData()
        if name == "EB-PHATE":
            return EBData()
        if name == "EB-PCA":
            return EBData("pcs", max_dim=max_dim, val_size=val_size)

        if name == "CHAFFER":
            return ChafferData()
        if name == "CHAFFER-PHATE":
            return ChafferData(embedding_name="phate")
        if name == "CHAFFER-PHATE-SMALL":
            return ChafferData(embedding_name="phate-small")

        if name == "WNV":
            return WNVData()
        if name == "WNV-PHATE":
            return WNVData(embedding_name="phate")

        if name == "NOONAN":
            return NoonanData()
        if name == "NOONAN-PCA":
            return NoonanData(max_dim=max_dim)

        if name == "WOT-FDL":
            return SchiebingerData("original_embedding")
        if name == "WOT-PHATE":
            return SchiebingerData("phate")
        if name == "WOT-PCA":
            return SchiebingerData("pcs", max_dim=max_dim)

        # If none of the above, we assume a path to a .npz file is supplied
        return CustomData(cfg)


class CustomData(SCData):
    def __init__(self, cfg):
        super().__init__()

        self.embedding_name = cfg['embedding_name']
        self.load(cfg['name'], cfg['max_dim'])

    def load(self, data_file, max_dim):
        self.data_dict = np.load(data_file, allow_pickle=True)
        self.labels = self.data_dict["sample_labels"]
        if self.embedding_name not in self.data_dict.keys():
            raise ValueError("Unknown embedding name %s" % self.embedding_name)
        embedding = self.data_dict[self.embedding_name]
        scaler = StandardScaler()
        scaler.fit(embedding)
        self.ncells = embedding.shape[0]
        assert self.labels.shape[0] == self.ncells
        # Scale so that embedding is normally distributed
        self.data = scaler.transform(embedding)

        delta_name = "delta_%s" % self.embedding_name
        if delta_name not in self.data_dict.keys(): 
            print("No velocity found for embedding %s skipping velocity" % self.embedding_name)
            self.use_velocity = False
        else:
            delta = self.data_dict[delta_name]
            assert delta.shape[0] == self.ncells
            # Normalize ignoring mean from embedding
            self.velocity = delta / scaler.scale_

        if max_dim is not None and self.data.shape[1] > max_dim:
            print("Warning: Clipping dimensionality to %d" % max_dim)
            self.data = self.data[:, :max_dim]
            if self.use_velocity:
                self.velocity = self.velocity[:, :max_dim]

    def has_velocity(self):
        return self.use_velocity

    def known_base_density(self):
        return False

    def get_data(self):
        return self.data

    def get_times(self):
        return self.labels

    def get_unique_times(self):
        return np.unique(self.labels)

    def get_velocity(self):
        return self.velocity

    def get_shape(self):
        return [self.data.shape[1]]

    def get_ncells(self):
        return self.ncells

    def leaveout_timepoint(self, tp):
        """ Takes a timepoint label to leaveout
        Alters data stored in object to leave out all data associated
        with that timepoint.
        """
        if tp < 0:
            raise RuntimeError("Cannot leaveout negative timepoint %d." % tp)
        mask = self.labels != tp
        print("Leaving out %d samples from sample %d" % (np.sum(~mask), tp))
        self.labels = self.labels[mask]
        self.data = self.data[mask]
        self.velocity = self.velocity[mask]
        self.ncells = np.sum(mask)

    def sample_index(self, n, label_subset):
        arr = np.arange(self.ncells)[self.labels == label_subset]
        return np.random.choice(arr, size=n)


class EBData(SCData):
    def __init__(
        self, embedding_name="phate", max_dim=None, val_size=0.1, use_velocity=True, version=5, 
    ):
        super().__init__()
        self.embedding_name = embedding_name
        self.use_velocity = use_velocity
        if version == 5:
            data_file = "../data/eb_velocity_v5.npz"
        else:
            raise ValueError("Unknown Version number")
        self.val_size = val_size
        self.load(data_file, max_dim)
        

    def load(self, data_file, max_dim):
        self.data_dict = np.load(data_file, allow_pickle=True)
        self.labels = self.data_dict["sample_labels"]
        if self.embedding_name not in self.data_dict.keys():
            raise ValueError("Unknown embedding name %s" % self.embedding_name)
        embedding = self.data_dict[self.embedding_name]
        scaler = StandardScaler()
        scaler.fit(embedding)
        self.ncells = embedding.shape[0]
        assert self.labels.shape[0] == self.ncells
        # Scale so that embedding is normally distributed
        self.data = scaler.transform(embedding)
        if self.has_velocity() and self.use_velocity:
            if self.embedding_name == "pcs":
                delta = self.data_dict["pcs_delta"]
            elif self.embedding_name == "phate":
                delta = self.data_dict["delta_embedding"]
            else:
                raise NotImplementedError("rna velocity must use phate")
            assert delta.shape[0] == self.ncells
            # Ignore mean from embedding
            self.velocity = delta / scaler.scale_

        if max_dim is not None and self.data.shape[1] > max_dim:
            print("Warning: Clipping dimensionality to %d" % max_dim)
            self.data = self.data[:, :max_dim]
            if self.has_velocity() and self.use_velocity:
                self.velocity = self.velocity[:, :max_dim]

        self._split_train_val()
        
    def _split_train_val(self):
        times = self.get_unique_times()
        min_n = min([ list(self.labels).count(label) for label in times ])
        self.val_n = int(min_n * self.val_size)
        shape = self.get_shape()
        self.train_data =  np.vstack([ self.data[self.labels == label][:-self.val_n]  for label in times ])
        self.val_data =  np.vstack([ self.data[self.labels == label][-self.val_n:]  for label in times ])
        self.train_labels = np.vstack([ self.labels[self.labels == label][:-self.val_n].reshape(-1, 1)  for label in times ]).flatten()
        self.val_labels = np.vstack([ self.labels[self.labels == label][-self.val_n:].reshape(-1, 1) for label in times ]).flatten()

    def has_velocity(self):
        return True

    def known_base_density(self):
        return False

    def get_data(self, type='train'):
        if type == 'train':
            return self.train_data
        elif type == 'val':
            return self.val_data
        else:
            return self.data

    def get_times(self, type='train'):
        if type == 'train':
            return self.train_labels
        elif type == 'val':
            return self.val_labels
        else:
            return self.labels

    def get_unique_times(self):
        return np.unique(self.labels)

    def get_velocity(self):
        return self.velocity

    def get_shape(self):
        return [self.data.shape[1]]

    def get_ncells(self, type='train'):
        if type == 'train':
            return len(self.train_labels)
        elif type == 'val':
            return len(self.val_labels)
        else:
            return self.ncells

    def leaveout_timepoint(self, tp):
        """ Takes a timepoint label to leaveout
        Alters data stored in object to leave out all data associated
        with that timepoint.
        """
        if tp < 0:
            raise RuntimeError("Cannot leaveout negative timepoint %d." % tp)
        mask = self.labels != tp
        print("Leaving out %d samples from sample %d" % (np.sum(~mask), tp))
        self.labels = self.labels[mask]
        self.data = self.data[mask]
        self.velocity = self.velocity[mask]
        self.ncells = np.sum(mask)

    def sample_index(self, n, label_subset, type='train'):
        if type == 'train':
            arr = np.arange(len(self.train_labels))[self.train_labels == label_subset]
        elif type == 'val':
            arr = np.arange(len(self.val_labels))[self.val_labels == label_subset]
        else:
            arr = np.arange(self.ncells)[self.labels == label_subset]
        return np.random.choice(arr, size=n)

class ChafferData(EBData):
    def __init__(self, embedding_name="pcs", max_dim=6, use_velocity=False, version=2):
        self.embedding_name = embedding_name
        self.use_velocity = use_velocity
        if version == 2:
            data_file = "../data/chaffer_v2.npz"
        else:
            raise ValueError("Unknown Version number")
        self.load(data_file, max_dim)


class WNVData(EBData):
    def __init__(self, embedding_name="pcs", max_dim=6, use_velocity=False, version=1):
        self.embedding_name = embedding_name
        self.use_velocity = use_velocity
        if version == 1:
            data_file = "../data/wnv_v1.npz"
        else:
            raise ValueError("Unknown Version number")
        self.load(data_file, max_dim)


class NoonanData(EBData):
    def __init__(self, embedding_name="pca", max_dim=2, use_velocity=True, version=1):
        self.embedding_name = embedding_name
        self.use_velocity = use_velocity
        if version == 1:
            data_file = "../data/noonan_v1.npz"
            validation_data_file = "../data/noonan_val_v1.npz"
        else:
            raise ValueError("Unknown Version number")
        self.load(data_file, validation_data_file, max_dim)

    def has_validation_samples(self):
        return True

    def load(self, data_file, val_data_file, max_dim):
        self.data_dict = np.load(data_file, allow_pickle=True)
        self.val_data_dict = np.load(val_data_file, allow_pickle=True)

        self.labels = self.data_dict["sample_labels"]
        if ("X_%s" % self.embedding_name) not in self.data_dict.keys():
            raise ValueError("Unknown embedding name %s" % self.embedding_name)
        embedding = self.data_dict["X_%s" % self.embedding_name]
        scaler = StandardScaler()
        scaler.fit(embedding)
        self.ncells = embedding.shape[0]
        assert self.labels.shape[0] == self.ncells
        # Scale so that embedding is normally distributed
        self.data = scaler.transform(embedding)
        self.val_data = scaler.transform(
            self.val_data_dict["X_%s" % self.embedding_name]
        )
        self.val_labels = self.val_data_dict["sample_labels"]

        if self.has_velocity() and self.use_velocity:
            delta = self.data_dict["velocity_%s" % self.embedding_name]
            assert delta.shape[0] == self.ncells
            self.velocity = delta / scaler.scale_

            # self.velocity = scaler.transform(delta)
            if max_dim is not None and self.velocity.shape[1] > max_dim:
                self.velocity = self.velocity[:, :max_dim]

        # TODO remove this and allow for other dimensional datasets besides 2
        if max_dim is not None and self.data.shape[1] > max_dim:
            print("Warning: Clipping dimensionality to %d" % max_dim)
            self.data = self.data[:, :max_dim]
            self.val_data = self.val_data[:, :max_dim]


class SchiebingerData(EBData):
    """ Load data from Schiebinger et al. (Cell 2019)
    39 timepoint samples. original embedding uses a force directed layout
    """

    def __init__(self, embedding_name="phate", max_dim=None):
        assert embedding_name in set(["phate", "original_embedding", "pcs"])
        data_file = "../data/schiebinger.npz"
        self.embedding_name = embedding_name
        self.load(data_file, max_dim)

    def has_velocity(self):
        return False

    def known_base_density(self):
        return False


class ClarkData(EBData):
    def __init__(self, embedding_name="original_embedding", max_dim=None):
        assert embedding_name in set(["phate", "original_embedding", "pcs"])
        data_file = "../data/clark.npz"
        self.embedding_name = embedding_name
        self.load(data_file, max_dim)

    def has_velocity(self):
        return False

    def known_base_density(self):
        return False

    def load(self, data_file, max_dim):
        """ Separates replicates from main trajectory.

        TODO: There may be some leakage of replicates by the phate coordinates used
        """
        super().load(data_file, max_dim)
        p = re.compile("rep2")
        replicates = [s for s in np.unique(self.labels) if p.search(s) is not None]
        replicate_indices = np.array([(s in replicates) for s in self.labels])
        self.data = self.data[~replicate_indices]
        self.labels = self.data_dict["times"][~replicate_indices]
        self.ncells = np.sum(~replicate_indices)

        # Mask out P cells
        mask = self.labels < 10
        self.data[mask]
        self.labels[mask]
        self.ncells = np.sum(mask)


class CircleTestData(EBData):
    """ Implements the circle dataset """

    def __init__(self):
        super().__init__()
        np.random.seed(42)
        n = 5000
        r1, r2, r3 = (0.4, 0.1, 0.1)

        self.labels = np.repeat(np.arange(3), n)
        eps = np.random.randn(3 * n)
        x = (self.labels * np.pi / 4) + np.pi / 4 + eps * r1
        x2d = np.array([np.cos(x), np.sin(x)]).T
        x2d += np.random.randn(*x2d.shape) * r2
        self.data = x2d
        self.ncells = self.data.shape[0]

        next2d = np.array([np.cos(x + 0.3), np.sin(x + 0.3)]).T
        next2d += np.random.randn(*next2d.shape) * r3
        self.velocity = next2d - x2d

    def has_velocity(self):
        return True


class CircleTestDataV2(EBData):
    """ Implements the circle dataset.

    But has an anlalytical base density and two timepoints instead of 3.
    """

    def __init__(self):
        super().__init__()
        np.random.seed(42)
        n = 5000
        self.r1, self.r2, self.r3 = (0.25, 0.1, 0.1)
        self.r1, self.r2, self.r3 = (0.4, 0.1, 0.1)

        self.labels = np.repeat(np.arange(2), n)
        theta = (self.labels * np.pi / 4) + np.pi / 2
        theta += np.random.randn(*theta.shape) * self.r1
        # Move set 0 to a weird place for verification
        # TODO remove
        # theta[self.labels == 0] += np.pi / 2
        r = (1 + np.random.randn(*theta.shape) * self.r2)[:, None]
        r = np.repeat(r, 2, axis=1)
        x2d = np.array([np.cos(theta), np.sin(theta)]).T * r
        # x2d[self.labels == 1] -= [0.7, 0.0]
        self.data = x2d
        self.ncells = self.data.shape[0]

        next2d = np.array([np.cos(theta + 0.3), np.sin(theta + 0.3)]).T * r
        next2d += np.random.randn(*next2d.shape) * self.r3
        self.velocity = next2d - x2d

    def base_density(self):
        def logprob(z):
            r = torch.sqrt(torch.sum(z.pow(2), 1))
            theta = torch.atan2(z[:, 0], z[:, 1])
            zp1 = (r - 1) / self.r2
            zp2 = theta - np.pi / 4
            zp2[zp2 > np.pi] -= 2 * np.pi
            zp2[zp2 < -np.pi] += 2 * np.pi
            zp2 = zp2 / self.r1
            # Find Quadrant
            logZ = -0.5 * math.log(2 * math.pi)
            z_polar = torch.stack([zp1, zp2], 1)
            return torch.sum(logZ - z_polar.pow(2) / 2, 1, keepdim=True)

        return logprob

    def base_sample(self):
        def f(*args, **kwargs):
            sample = torch.randn(*args, **kwargs)
            theta = sample[:, 0] * self.r1 + np.pi / 4
            r = (sample[:, 1] * self.r2 + 1)[:, None]
            s = torch.stack([torch.cos(theta), torch.sin(theta)], 1) * r
            return s

        return f

    def has_velocity(self):
        return True


class CircleTestDataV3(EBData):
    """ Implements the curvy tree dataset.

    Has an analytical base density and two timepoints instead of 3.
    Where the base distribution is a half-gaussian at theta=0 and the end 
    distribution is a half-gaussian at theta=2*pi. Both truncated below y=0. 
    this is to experiment with the standard deviation of theta to see if
    we can learn a flow along the circle instead of across it.
    The hope is that the default flow is across the circle where we can
    regularize it towards density.
    """

    def __init__(self):
        super().__init__()
        np.random.seed(42)
        n = 5000
        self.r1, self.r2, self.r3 = (0.25, 0.1, 0.1)
        self.r1, self.r2, self.r3 = (0.4, 0.1, 0.1)
        self.r1, self.r2, self.r3 = (0.5, 0.1, 0.1)

        self.labels = np.repeat(np.arange(2), n)
        theta = (self.labels * np.pi / 2) + np.pi / 2
        # theta = (self.labels * np.pi / 4) + np.pi / 2
        theta += np.random.randn(*theta.shape) * self.r1
        # Move set 0 to a weird place for verification
        # TODO remove
        # theta[self.labels == 0] += np.pi / 2
        theta[self.labels == 0] += np.random.randn(*theta.shape)[self.labels == 0] * 2
        theta[theta < 0] *= -1
        theta[theta > np.pi] = 2 * np.pi - theta[theta > np.pi]
        r = (1 + np.random.randn(*theta.shape) * self.r2)[:, None]
        r = np.repeat(r, 2, axis=1)
        x2d = np.array([np.cos(theta), np.sin(theta)]).T * r
        # x2d[self.labels == 1] -= [0.7, 0.0]
        # x2d[x2d[:, 1] < 0] *= [1, -1]
        self.data = x2d
        self.ncells = self.data.shape[0]

        next2d = np.array([np.cos(theta + 0.3), np.sin(theta + 0.3)]).T * r
        # next2d += np.random.randn(*next2d.shape) * self.r3
        self.velocity = next2d - x2d

    def base_density(self):
        def logprob(z):
            # I no longer understand how this function works, but it looks right
            r = torch.sqrt(torch.sum(z.pow(2), 1))
            theta = torch.atan2(z[:, 0], -z[:, 1])
            zp1 = (r - 1) / self.r2
            zp2 = theta - np.pi / 2
            # zp2 = (theta - np.pi / 4)
            zp2[zp2 > np.pi] -= 2 * np.pi
            zp2[zp2 < -np.pi] += 2 * np.pi
            zp2 = zp2 / self.r1
            # Find Quadrant
            logZ = -0.5 * math.log(2 * math.pi)
            z_polar = torch.stack([zp1, zp2], 1)
            to_return = torch.sum(logZ - z_polar.pow(2) / 2, 1, keepdim=True)
            to_return[zp2 < 0] += 20 * zp2[zp2 < 0][:, None]
            # to_return[zp2 >= 0] -= 0    # Multiply in log space?
            # to_return[zp2 < 0] += 50
            # to_return[zp2 >= 0] -= 50    # Multiply in log space?
            return to_return

        return logprob

    def known_base_density(self):
        return True

    def base_sample(self):
        def f(*args, **kwargs):
            sample = torch.randn(*args, **kwargs)
            theta = sample[:, 0] * self.r1
            r = (sample[:, 1] * self.r2 + 1)[:, None]
            s = torch.stack([torch.cos(theta), torch.sin(theta)], 1) * r
            s[s[:, 1] < 0] *= torch.tensor([1, -1], dtype=torch.float32)[None, :]
            return s

        return f

    def has_velocity(self):
        return True


class CircleTestDataV4(CircleTestDataV3):
    """ BROKEN """

    def __init__(self):
        super().__init__()
        # self.velocity[:, 0] = -self.velocity[:, 0]
        # self.data[:, 0] = -self.data[:, 0]
        raise NotImplementedError("This version of the dataset is broken")

    def base_density(self):
        def logprob(z):
            z = z * torch.tensor([-1, 1], dtype=torch.float32).to(z)[None, :]
            # I no longer understand how this function works, but it looks right
            r = torch.sqrt(torch.sum(z.pow(2), 1))
            theta = torch.atan2(z[:, 0], -z[:, 1])
            zp1 = (r - 1) / self.r2
            zp2 = theta - np.pi / 2
            # zp2 = (theta - np.pi / 4)
            zp2[zp2 > np.pi] -= 2 * np.pi
            zp2[zp2 < -np.pi] += 2 * np.pi
            zp2 = zp2 / self.r1
            # Find Quadrant
            logZ = -0.5 * math.log(2 * math.pi)
            z_polar = torch.stack([zp1, zp2], 1)
            to_return = torch.sum(logZ - z_polar.pow(2) / 2, 1, keepdim=True)
            to_return[zp2 < 0] += 20 * zp2[zp2 < 0][:, None]
            # to_return[zp2 >= 0] -= 0    # Multiply in log space?
            # to_return[zp2 < 0] += 50
            # to_return[zp2 >= 0] -= 50    # Multiply in log space?
            return to_return

        return logprob

    def known_base_density(self):
        return True

    def base_sample(self):
        def f(*args, **kwargs):
            sample = torch.randn(*args, **kwargs)
            theta = sample[:, 0] * self.r1
            r = (sample[:, 1] * self.r2 + 1)[:, None]
            s = torch.stack([torch.cos(theta), torch.sin(theta)], 1) * r
            s[s[:, 1] < 0] *= torch.tensor([1, -1], dtype=torch.float32)[None, :]
            s = s * torch.tensor([-1, 1], dtype=torch.float32)[None, :]
            return s

        return f


def interpolate_with_ot(p0, p1, tmap, interp_frac, size):
    """
    Interpolate between p0 and p1 at fraction t_interpolate knowing a transport map from p0 to p1
    Parameters
    ----------
    p0 : 2-D array
        The genes of each cell in the source population
    p1 : 2-D array
        The genes of each cell in the destination population
    tmap : 2-D array
        A transport map from p0 to p1
    t_interpolate : float
        The fraction at which to interpolate
    size : int
        The number of cells in the interpolated population
    Returns
    -------
    p05 : 2-D array
        An interpolated population of 'size' cells
    """
    p0 = p0.toarray() if scipy.sparse.isspmatrix(p0) else p0
    p1 = p1.toarray() if scipy.sparse.isspmatrix(p1) else p1
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    print(p0.shape, p1.shape)
    tmap = np.asarray(tmap, dtype=np.float64)
    if p0.shape[1] != p1.shape[1]:
        raise ValueError("Unable to interpolate. Number of genes do not match")
    if p0.shape[0] != tmap.shape[0] or p1.shape[0] != tmap.shape[1]:
        raise ValueError(
            "Unable to interpolate. Tmap size is {}, expected {}".format(
                tmap.shape, (len(p0), len(p1))
            )
        )
    I = len(p0)
    J = len(p1)
    # Assume growth is exponential and retrieve growth rate at t_interpolate
    p = tmap / np.power(tmap.sum(axis=0), 1.0 - interp_frac)
    p = p.flatten(order="C")
    p = p / p.sum()
    choices = np.random.choice(I * J, p=p, size=size)
    return np.asarray(
        [p0[i // J] * (1 - interp_frac) + p1[i % J] * interp_frac for i in choices],
        dtype=np.float64,
    )


class TreeTestData(CircleTestDataV3):
    def __init__(self):
        np.random.seed(42)
        n = 5000
        self.r1, self.r2, self.r3 = (0.5, 0.1, 0.1)
        self.labels = np.repeat([0, 2], n)

        data = np.abs(np.random.randn(n * 2) * 0.5 / np.pi)
        data[self.labels == 2] = 1 - data[self.labels == 2]
        # print(data)

        # McCann interpolant / barycenter interpolation
        import ot

        gamma = ot.emd_1d(data[self.labels == 0], data[self.labels == 2])
        ninterp = 5000
        i05 = interpolate_with_ot(
            data[self.labels == 0][:, np.newaxis],
            data[self.labels == 2][:, np.newaxis],
            gamma,
            0.5,
            ninterp,
        )
        data = np.concatenate([data, i05.flatten()])
        self.labels = np.concatenate([self.labels, np.ones(n)])
        theta = data * np.pi  # transform to along the circle

        r = (1 + np.random.randn(*theta.shape) * self.r2)[:, None]
        r = np.repeat(r, 2, axis=1)
        x2d = np.array([np.cos(theta), np.sin(theta)]).T * r

        mask = np.random.rand(x2d.shape[0]) > 0.5
        mask *= x2d[:, 0] < 0
        x2d[mask] = [[0, 2]] + [[1, -1]] * x2d[mask]

        # x2d[self.labels == 1] -= [0.7, 0.0]
        # x2d[x2d[:, 1] < 0] *= [1, -1]
        self.data = x2d
        self.ncells = self.data.shape[0]

        next2d = np.array([np.cos(theta + 0.3), np.sin(theta + 0.3)]).T * r
        next2d[mask] = [[0, 2]] + [[1, -1]] * next2d[mask]
        # next2d += np.random.randn(*next2d.shape) * self.r3
        self.velocity = next2d - x2d

        # Mask out timepoint zero
        mask = self.labels != 0
        self.labels = self.labels[mask]
        self.labels -= 1
        self.data = self.data[mask]
        self.velocity = self.velocity[mask]
        self.ncells = self.labels.shape[0]

    def get_paths(self, n=5000, n_steps=3):
        # Only 3 steps are supported at this time.
        assert n_steps == 3
        np.random.seed(42)
        self.r1, self.r2, self.r3 = (0.5, 0.1, 0.1)
        labels = np.repeat([0, 2], n)

        data = np.abs(np.random.randn(n * 2) * 0.5 / np.pi)
        data[labels == 2] = 1 - data[labels == 2]
        # print(data)

        # McCann interpolant / barycenter interpolation
        import ot

        gamma = ot.emd_1d(data[labels == 0], data[labels == 2])
        ninterp = 5000
        i05 = interpolate_with_ot(
            data[labels == 0][:, np.newaxis],
            data[labels == 2][:, np.newaxis],
            gamma,
            0.5,
            ninterp,
        )
        # data = data.reshape(-1, 2)
        data = np.stack([data[labels == 0], i05.flatten(), data[labels == 2]], axis=-1)

        theta = data * np.pi  # transform to along the circle

        r = (1 + np.random.randn(n) * self.r2)[:, None, None]

        x2d = np.stack([np.cos(theta), np.sin(theta)], axis=-1) * r
        # mask = (r > 1.0)
        # TODO these reference paths could be improved to include better routing
        # along the manifold. Right now they are calculated using 1d and are just lifted into
        # 2d along the same radius. Trouble comes when the branch for the tree gets
        # Flipped over y=1, this gives opposite of expected radiuses.
        # Furthermore, 2d Transport is no longer the same as 1d when we have gaussian
        # Noise along the manifold.
        #
        # Right now they are good enough for our purposes, and making them better will only
        # improve how TrajectoryNet looks.
        """
        import optimal_transport.emd as emd
        _, log = emd.earth_mover_distance(x2d[:,0], x2d[:,1], return_matrix=True)
        print(np.where(log['G'] > 1e-8))
        path = np.stack([x2d[:,0], x2d[np.where(log['G'] > 1e-8)[1],1]])
        path = np.swapaxes(path, 0,1)
        import matplotlib.pyplot as plt
        #plt.hist(log['G'].flatten())
        fig, axes = plt.subplots(1,2,figsize=(20,10))

        for p in path[:1000]:
            axes[0].plot(p[:,0], p[:,1])
        for p in x2d[:1000,:2]:
            axes[1].plot(p[:,0], p[:,1])
        plt.show()
        exit()
        """
        mask = np.random.rand(*x2d.shape[:2]) > 0.5
        mask *= x2d[:, :, 0] < 0
        x2d[mask] = [[0, 2]] + [[1, -1]] * x2d[mask]
        x2d = x2d.reshape(n, n_steps, 2)
        return x2d
        # Samples x Time x Dimension
        # return x2d


class CircleTestDataV5(TreeTestData):
    """ This builds on version 3 to include a better middle timepoint.
    Where instead of being parametrically defined, the middle timepoint is
    defined in terms of the interpolant between the first and last
    timepoints along the manifold.

    This is a useful thing to relate to in terms of transport along the manifold.
    """

    def __init__(self):
        np.random.seed(42)
        n = 5000
        self.r1, self.r2, self.r3 = (0.5, 0.1, 0.1)
        self.labels = np.repeat([0, 2], n)

        data = np.abs(np.random.randn(n * 2) * 0.5 / np.pi)
        data[self.labels == 2] = 1 - data[self.labels == 2]
        # print(data)

        # McCann interpolant / barycenter interpolation
        import ot

        gamma = ot.emd_1d(data[self.labels == 0], data[self.labels == 2])
        ninterp = 5000
        i05 = interpolate_with_ot(
            data[self.labels == 0][:, np.newaxis],
            data[self.labels == 2][:, np.newaxis],
            gamma,
            0.5,
            ninterp,
        )
        data = np.concatenate([data, i05.flatten()])
        self.labels = np.concatenate([self.labels, np.ones(n)])
        theta = data * np.pi  # transform to along the circle

        r = (1 + np.random.randn(*theta.shape) * self.r2)[:, None]
        r = np.repeat(r, 2, axis=1)
        x2d = np.array([np.cos(theta), np.sin(theta)]).T * r

        ##########################
        # ONLY CHANGE FROM ABOVE #
        mask = np.random.rand(x2d.shape[0]) > 1.0
        ##########################

        mask *= x2d[:, 0] < 0
        x2d[mask] = [[0, 2]] + [[1, -1]] * x2d[mask]

        # x2d[self.labels == 1] -= [0.7, 0.0]
        # x2d[x2d[:, 1] < 0] *= [1, -1]
        self.data = x2d
        self.ncells = self.data.shape[0]

        next2d = np.array([np.cos(theta + 0.3), np.sin(theta + 0.3)]).T * r
        next2d[mask] = [[0, 2]] + [[1, -1]] * next2d[mask]
        # next2d += np.random.randn(*next2d.shape) * self.r3
        self.velocity = next2d - x2d

        # Mask out timepoint zero
        mask = self.labels != 0
        self.labels = self.labels[mask]
        self.labels -= 1
        self.data = self.data[mask]
        self.velocity = self.velocity[mask]
        self.ncells = self.labels.shape[0]

    def get_paths(self, n=5000, n_steps=3):
        # Only 3 steps are supported at this time.
        assert n_steps == 3
        np.random.seed(42)
        self.r1, self.r2, self.r3 = (0.5, 0.1, 0.1)
        labels = np.repeat([0, 2], n)

        data = np.abs(np.random.randn(n * 2) * 0.5 / np.pi)
        data[labels == 2] = 1 - data[labels == 2]
        # print(data)

        # McCann interpolant / barycenter interpolation
        import ot

        gamma = ot.emd_1d(data[labels == 0], data[labels == 2])
        ninterp = 5000
        i05 = interpolate_with_ot(
            data[labels == 0][:, np.newaxis],
            data[labels == 2][:, np.newaxis],
            gamma,
            0.5,
            ninterp,
        )
        # data = data.reshape(-1, 2)
        data = np.stack([data[labels == 0], i05.flatten(), data[labels == 2]], axis=-1)

        theta = data * np.pi  # transform to along the circle

        r = (1 + np.random.randn(n) * self.r2)[:, None, None]

        x2d = np.stack([np.cos(theta), np.sin(theta)], axis=-1) * r
        return x2d


class CycleDataset(TreeTestData):
    """ The idea here is that the distribution does not change, 
    but there is movement around the circle over time.
    
    First we define a rotation speed with a uniform distribution around the circle. 
    
    We generate this by taking a uniform distribution then rotating it 1/4 way around the circle.
    
    The interpolation is then 1/8 of the way around the circle. We need a new evaluation mechanism to
    be able to handle this case, as distribution level, all are approximately zero difference. 
    """

    def __init__(self, shift=0.1, r_std=0.1):
        np.random.seed(42)
        n = 5000
        self.shift = shift
        self.r_std = r_std
        data = np.random.rand(n)
        data = np.concatenate([data, data + shift, data + 2 * shift])
        r = np.tile(np.ones(n) + np.random.randn(n) * self.r_std, 3)[:, np.newaxis]
        self.labels = np.repeat(np.arange(2), n)
        theta = data * 2 * np.pi
        x2d = np.array([np.cos(theta), np.sin(theta)]).T * r
        self.data = x2d[n:]
        self.old_data = x2d[:n]
        next_theta = theta + 2 * np.pi * shift * 0.001
        next2d = np.array([np.cos(next_theta), np.sin(next_theta)]).T * r
        self.velocity = ((next2d - x2d) * 1000)[n:] * 2
        self.ncells = 2 * n

    def get_paths(self, n=5000, n_steps=3):
        # Only 3 steps are supported at this time.
        assert n_steps == 3
        shift = self.shift
        np.random.seed(42)
        data = np.random.rand(n)
        data = np.stack([data, data + shift, data + 2 * shift], axis=0)
        r = (np.ones(n) + np.random.randn(n) * self.r_std)[np.newaxis, :, np.newaxis]
        theta = data * 2 * np.pi
        x2d = np.stack([np.cos(theta), np.sin(theta)], axis=-1) * r
        x2d = np.swapaxes(x2d, 0, 1)
        # Samples x Time x Dimension
        return x2d

    def base_density(self):
        # It is OK if this is only proportional to the true distribution
        # As long as it is relatively close for scaling purposes
        def logprob(z):
            r = torch.sqrt(torch.sum(z.pow(2), 1))
            zp1 = (r - 1) / self.r_std
            logZ = -0.5 * math.log(2 * math.pi * self.r_std * self.r_std)
            to_return = logZ - zp1.pow(2) / 2
            # I don't know why this correction factor works, but it seems to integrate to 1 now.
            return (to_return - math.log(2 * np.pi))[:, np.newaxis]

        return logprob

    def known_base_density(self):
        return True

    def base_sample(self):
        def f(*args, **kwargs):
            sample = torch.randn(*args, **kwargs)
            sample_uniform = torch.rand(*args, **kwargs)
            theta = sample_uniform[:, 0] * 2 * np.pi
            r = (sample[:, 0] * self.r_std + 1)[:, None]
            s = torch.stack([torch.cos(theta), torch.sin(theta)], 1) * r
            return s

        return f


class SklearnData(SCData):
    def __init__(self, name="moons", n_samples=10000):
        import sklearn.datasets

        self.name = name
        # From sklearn auto_examples/cluster/plot_cluster_comparison
        seed = 42
        np.random.seed(seed)
        if name == "circles":
            self.data, _ = sklearn.datasets.make_circles(
                n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
            )
            self.data *= 3.5
        elif name == "moons":
            self.data, _ = sklearn.datasets.make_moons(
                n_samples=n_samples, noise=0.05, random_state=seed
            )
            self.data *= 2
            self.data[:, 0] -= 1
        elif name == "blobs":
            self.data, _ = sklearn.datasets.make_blobs(n_samples=n_samples)
        elif name == "scurve":
            self.data, _ = sklearn.datasets.make_s_curve(
                n_samples=n_samples, noise=0.05, random_state=seed
            )
            self.data = np.vstack([self.data[:, 0], self.data[:, 2]]).T
            self.data *= 1.5
        else:
            raise NotImplementedError("Unknown dataset name %s" % name)

    def get_times(self):
        return np.repeat([0], self.data.shape[0])

    def get_unique_times(self):
        return [0]

    def has_velocity(self):
        return False

    def known_base_density(self):
        return True

    def get_data(self):
        return self.data

    def get_shape(self):
        return [self.data.shape[1]]

    def get_ncells(self):
        return self.data.shape[0]

    def base_density(self):
        def standard_normal_logprob(z):
            logZ = -0.5 * math.log(2 * math.pi)
            return torch.sum(logZ - z.pow(2) / 2, 1, keepdim=True)

        return standard_normal_logprob

    def base_sample(self):
        return torch.randn

    def sample_index(self, n, label_subset):
        arr = np.arange(self.get_ncells())[self.get_times() == label_subset]
        return np.random.choice(arr, size=n)


if __name__ == "__main__":
    data = WNVData("pcs", max_dim=20)
