import logging
import numpy as np
from sekupy.preprocessing.mne.base import MneTransformer

logger = logging.getLogger(__name__)


class Filter(MneTransformer):

    def __init__(self, l_freq=None, h_freq=None, method='fir', **kwargs):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method
        self._filter_kwargs = kwargs
        MneTransformer.__init__(self, name='filter', l_freq=l_freq, h_freq=h_freq)

    def transform(self, ds):
        logger.info('Filtering: l_freq=%s h_freq=%s', self.l_freq, self.h_freq)
        ds.filter(l_freq=self.l_freq, h_freq=self.h_freq,
                  method=self.method, **self._filter_kwargs)
        return ds


class NotchFilter(MneTransformer):

    def __init__(self, freqs, method='fir', **kwargs):
        self.freqs = freqs
        self.method = method
        self._filter_kwargs = kwargs
        MneTransformer.__init__(self, name='notch_filter', freqs=freqs)

    def transform(self, ds):
        logger.info('Notch filtering at %s Hz', str(self.freqs))
        ds.notch_filter(freqs=self.freqs, method=self.method,
                        **self._filter_kwargs)
        return ds


class Resample(MneTransformer):

    def __init__(self, sfreq, **kwargs):
        self.sfreq = sfreq
        self._resample_kwargs = kwargs
        MneTransformer.__init__(self, name='resample', sfreq=sfreq)

    def transform(self, ds):
        logger.info('Resampling to %s Hz', str(self.sfreq))
        ds.resample(sfreq=self.sfreq, **self._resample_kwargs)
        return ds


class DropChannels(MneTransformer):

    def __init__(self, ch_names, **kwargs):
        self.ch_names = list(ch_names)
        MneTransformer.__init__(self, name='drop_channels', ch_names=ch_names)

    def transform(self, ds):
        to_drop = [ch for ch in self.ch_names if ch in ds.ch_names]
        logger.info('Dropping channels: %s', str(to_drop))
        if to_drop:
            ds.drop_channels(to_drop)
        return ds


class SetMontage(MneTransformer):

    def __init__(self, montage, match_case=True, **kwargs):
        self.montage = montage
        self.match_case = match_case
        self._montage_kwargs = kwargs
        MneTransformer.__init__(self, name='set_montage', montage=str(montage))

    def transform(self, ds):
        logger.info('Setting montage: %s', str(self.montage))
        ds.set_montage(self.montage, match_case=self.match_case,
                       **self._montage_kwargs)
        return ds


class RemoveBadChannels(MneTransformer):
    """Detect bad channels by flagging outliers in per-channel standard deviation."""

    def __init__(self, threshold=5.0, picks=None, **kwargs):
        self.threshold = threshold
        self.picks = picks
        MneTransformer.__init__(self, name='remove_bad_channels',
                                threshold=threshold)

    def transform(self, ds):
        import mne
        from scipy import stats

        picks = mne.pick_types(ds.info, eeg=True, meg=True) \
            if self.picks is None else self.picks
        data = ds.get_data(picks=picks)

        channel_std = data.std(axis=1)
        z_scores = np.abs(stats.zscore(channel_std))
        bad_mask = (z_scores > self.threshold) | (channel_std < 1e-10)

        ch_names = np.array(ds.ch_names)[picks]
        bad_chs = ch_names[bad_mask].tolist()
        ds.info['bads'] = list(set(ds.info['bads'] + bad_chs))
        logger.info('Bad channels detected: %s', str(bad_chs))
        return ds


class Crop(MneTransformer):

    def __init__(self, tmin=None, tmax=None, **kwargs):
        self.tmin = tmin
        self.tmax = tmax
        MneTransformer.__init__(self, name='crop', tmin=tmin, tmax=tmax)

    def transform(self, ds):
        logger.info('Cropping: tmin=%s tmax=%s', self.tmin, self.tmax)
        ds.crop(tmin=self.tmin, tmax=self.tmax)
        return ds


class Interpolate(MneTransformer):

    def __init__(self, reset_bads=True, **kwargs):
        self.reset_bads = reset_bads
        self._interp_kwargs = kwargs
        MneTransformer.__init__(self, name='interpolate', reset_bads=reset_bads)

    def transform(self, ds):
        logger.info('Interpolating bad channels: %s', str(ds.info['bads']))
        ds.interpolate_bads(reset_bads=self.reset_bads, **self._interp_kwargs)
        return ds
