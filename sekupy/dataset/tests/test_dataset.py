"""Regression tests for the Dataset object model (see sekupy/dataset/dataset.py,
flatten.py, zscore.py, detrend.py), covering the behavior the rest of the
codebase actually relies on: indexed selection keeps sa/fa aligned,
vstack/hstack semantics, the flatten/mask/unflatten brain-space round trip,
and chunk-wise zscore/detrend.
"""
import numpy as np
import pytest

from sekupy.dataset.dataset import Dataset, vstack, hstack
from sekupy.dataset.flatten import flatten_dataset, flatten_array, unflatten_to_brain
from sekupy.dataset.zscore import zscore
from sekupy.dataset.detrend import poly_detrend


def _toy_dataset(n_samples=6, n_features=4):
    samples = np.arange(n_samples * n_features, dtype=float).reshape(n_samples, n_features)
    sa = {
        'targets': np.array((['a', 'b'] * n_samples)[:n_samples]),
        'chunks': np.arange(n_samples) % 2,
    }
    fa = {'roi': np.arange(n_features)}
    a = {'note': 'toy'}
    return Dataset(samples, sa=sa, fa=fa, a=a)


class TestConstruction:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            Dataset(np.zeros((4, 3)), sa={'targets': np.zeros(3)})

    def test_sa_fa_attribute_access(self):
        ds = _toy_dataset()
        np.testing.assert_array_equal(ds.sa.targets, ds.sa['targets'])
        np.testing.assert_array_equal(ds.fa.roi, ds.fa['roi'])
        assert ds.a.note == 'toy'

    def test_post_construction_assignment_is_length_checked(self):
        ds = _toy_dataset()
        ds.fa['new_attr'] = np.arange(ds.shape[1])  # ok
        with pytest.raises(ValueError):
            ds.fa['bad_attr'] = np.arange(ds.shape[1] - 1)

    def test_dataset_attrs_are_unconstrained(self):
        ds = _toy_dataset()
        ds.a['anything'] = object()  # no length check on `a`

    def test_a_sa_fa_copy_preserves_attribute_access(self):
        # dict.copy() returns a plain dict on a subclass unless overridden;
        # analysis/base.py's _store_info relies on `ds.a.copy()` still
        # supporting attribute access (`.mapper`, `.imgaffine`, ...).
        ds = _toy_dataset()

        a_copy = ds.a.copy()
        sa_copy = ds.sa.copy()

        assert a_copy.note == ds.a['note']
        assert isinstance(a_copy, type(ds.a))
        np.testing.assert_array_equal(sa_copy.targets, ds.sa['targets'])
        assert isinstance(sa_copy, type(ds.sa))
        # and it's still length-checked
        with pytest.raises(ValueError):
            sa_copy['bad'] = np.arange(ds.shape[0] - 1)

    def test_mutable_a_values_not_shared_across_derived_datasets(self):
        # regression test: two datasets both derived (via copy/__getitem__)
        # from the same source used to share the same `prepro` list object,
        # so appending on one silently showed up on the other -- this
        # matters a lot for a session-scoped fixture reused across tests.
        ds = _toy_dataset()
        ds.a['history'] = ['loaded']

        copy_ds = ds.copy()
        copy_ds.a['history'].append('copied')

        sliced_ds = ds[ds.sa.chunks == 0]
        sliced_ds.a['history'].append('sliced')

        assert ds.a['history'] == ['loaded']
        assert copy_ds.a['history'] == ['loaded', 'copied']
        assert sliced_ds.a['history'] == ['loaded', 'sliced']


class TestGetitemAlignment:
    def test_boolean_sample_mask_keeps_sa_aligned(self):
        ds = _toy_dataset()
        mask = ds.sa.targets == 'a'

        sub = ds[mask]

        assert sub.shape == (mask.sum(), ds.shape[1])
        np.testing.assert_array_equal(sub.samples, ds.samples[mask])
        np.testing.assert_array_equal(sub.sa.targets, ds.sa.targets[mask])
        np.testing.assert_array_equal(sub.sa.chunks, ds.sa.chunks[mask])
        np.testing.assert_array_equal(sub.fa.roi, ds.fa.roi)

    def test_boolean_feature_mask_keeps_fa_aligned(self):
        ds = _toy_dataset()
        mask = ds.fa.roi % 2 == 0

        sub = ds[:, mask]

        assert sub.shape == (ds.shape[0], mask.sum())
        np.testing.assert_array_equal(sub.samples, ds.samples[:, mask])
        np.testing.assert_array_equal(sub.fa.roi, ds.fa.roi[mask])
        np.testing.assert_array_equal(sub.sa.targets, ds.sa.targets)

    def test_combined_sample_and_feature_selection(self):
        ds = _toy_dataset()
        smask = ds.sa.chunks == 0
        fmask = ds.fa.roi < 2

        sub = ds[smask, fmask]

        assert sub.shape == (smask.sum(), fmask.sum())
        np.testing.assert_array_equal(sub.samples, ds.samples[np.ix_(smask, fmask)])
        np.testing.assert_array_equal(sub.sa.chunks, ds.sa.chunks[smask])
        np.testing.assert_array_equal(sub.fa.roi, ds.fa.roi[fmask])

    def test_integer_sample_selection_keeps_sample_axis(self):
        ds = _toy_dataset()

        one = ds[0]

        assert one.shape == (1, ds.shape[1])
        np.testing.assert_array_equal(one.samples, ds.samples[[0]])
        np.testing.assert_array_equal(one.sa.targets, ds.sa.targets[[0]])

    def test_iteration_yields_one_sample_datasets(self):
        ds = _toy_dataset(n_samples=3)

        rows = list(ds)

        assert len(rows) == 3
        for i, row in enumerate(rows):
            assert row.shape == (1, ds.shape[1])
            np.testing.assert_array_equal(row.samples[0], ds.samples[i])


class TestStacking:
    def test_vstack_concatenates_samples_and_sa(self):
        ds1 = _toy_dataset(n_samples=3)
        ds2 = _toy_dataset(n_samples=3)

        stacked = vstack([ds1, ds2])

        assert stacked.shape == (6, ds1.shape[1])
        np.testing.assert_array_equal(stacked.samples, np.vstack([ds1.samples, ds2.samples]))
        np.testing.assert_array_equal(
            stacked.sa.targets, np.concatenate([ds1.sa.targets, ds2.sa.targets])
        )
        np.testing.assert_array_equal(stacked.fa.roi, ds1.fa.roi)

    def test_vstack_requires_matching_sa_keys(self):
        ds1 = _toy_dataset(n_samples=3)
        ds2 = _toy_dataset(n_samples=3)
        del ds2.sa['chunks']

        with pytest.raises(ValueError):
            vstack([ds1, ds2])

    def test_vstack_drops_dataset_attrs_by_default(self):
        ds1 = _toy_dataset(n_samples=3)
        ds2 = _toy_dataset(n_samples=3)
        ds2.a['note'] = 'different'

        stacked = vstack([ds1, ds2])

        assert 'note' not in stacked.a

    def test_hstack_concatenates_samples_and_fa(self):
        ds1 = _toy_dataset(n_features=4)
        ds2 = _toy_dataset(n_features=4)

        stacked = hstack([ds1, ds2])

        assert stacked.shape == (ds1.shape[0], 8)
        np.testing.assert_array_equal(stacked.samples, np.hstack([ds1.samples, ds2.samples]))


class TestFlattenMaskReverseRoundtrip:
    def test_flatten_mask_and_unflatten_recovers_voxel_positions(self):
        rng = np.random.RandomState(0)
        vol_shape = (2, 3, 4)
        n_samples = 5
        data = rng.rand(n_samples, *vol_shape)

        ds = Dataset(data)
        ds = flatten_dataset(ds, space='voxel_indices')

        assert ds.shape == (n_samples, np.prod(vol_shape))
        assert ds.fa.voxel_indices.shape == (np.prod(vol_shape), len(vol_shape))

        mask = rng.rand(*vol_shape) > 0.5
        flatmask = flatten_array(mask, ds)
        masked = ds[:, flatmask != 0]

        assert masked.shape[1] == mask.sum()

        stat = np.arange(1, masked.shape[1] + 1, dtype=float)
        brain = unflatten_to_brain(stat, masked)

        assert brain.shape == vol_shape
        coords = masked.fa.voxel_indices
        for i, (x, y, z) in enumerate(coords):
            assert brain[x, y, z] == stat[i]
        covered = np.zeros(vol_shape, dtype=bool)
        covered[tuple(coords.T)] = True
        assert np.all(brain[~covered] == 0)


class TestZScoreBehavior:
    def test_chunkwise_zscore_matches_manual_computation(self):
        rng = np.random.RandomState(1)
        samples = rng.rand(8, 3) * 10
        chunks = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        ds = Dataset(samples, sa={'chunks': chunks})

        out = zscore(ds, chunks_attr='chunks')

        expected = np.empty_like(samples)
        for c in np.unique(chunks):
            m = chunks == c
            expected[m] = (samples[m] - samples[m].mean(axis=0)) / samples[m].std(axis=0)

        np.testing.assert_allclose(out.samples, expected, rtol=1e-6)
        np.testing.assert_array_equal(ds.samples, samples)


class TestPolyDetrendBehavior:
    def test_polyord_zero_removes_the_chunkwise_mean(self):
        rng = np.random.RandomState(2)
        samples = rng.rand(8, 3) * 10 + 5
        chunks = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        ds = Dataset(samples, sa={'chunks': chunks})

        out = poly_detrend(ds, polyord=0, chunks_attr='chunks')

        for c in np.unique(chunks):
            m = chunks == c
            np.testing.assert_allclose(out.samples[m].mean(axis=0), 0, atol=1e-8)
