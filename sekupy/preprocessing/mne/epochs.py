import logging
from sekupy.preprocessing.mne.base import MneTransformer

logger = logging.getLogger(__name__)


class AutoReject(MneTransformer):

    def __init__(self, random_state=42, n_jobs=1, **kwargs):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._ar_kwargs = kwargs
        self.reject_log_ = None
        MneTransformer.__init__(self, name='autoreject',
                                random_state=random_state)

    def transform(self, ds):
        from autoreject import AutoReject as AR
        logger.info('Running AutoReject...')
        ar = AR(random_state=self.random_state, n_jobs=self.n_jobs,
                **self._ar_kwargs)
        ds_clean, self.reject_log_ = ar.fit_transform(ds, return_log=True)
        logger.info('AutoReject dropped %d epochs',
                    self.reject_log_.bad_epochs.sum())
        return ds_clean


class Epoching(MneTransformer):
    """Create epochs from a Raw object.

    Parameters
    ----------
    kind : {'events', 'fixed_length'}
        ``'events'`` uses :func:`mne.Epochs` with a supplied events array.
        ``'fixed_length'`` uses :func:`mne.make_fixed_length_epochs`.
    events : array-like, optional
        Events array required when ``kind='events'``.
    event_id : dict or int, optional
        Event IDs passed to :func:`mne.Epochs`.
    tmin, tmax : float
        Epoch time limits (used by both kinds).
    duration : float
        Epoch duration in seconds for ``kind='fixed_length'``.
    overlap : float
        Overlap in seconds between consecutive fixed-length epochs.
    baseline : tuple or None
        Baseline correction window.
    picks : array-like or None
        Channel selection.
    preload : bool
        Load data into memory immediately.
    reject_by_annotation : bool
        Skip annotated bad segments (``kind='fixed_length'`` only).
    """

    def __init__(self, kind='events', events=None, event_id=None,
                 tmin=-0.2, tmax=1.0, duration=1.0, overlap=0.0,
                 baseline=(None, 0), picks=None, preload=True,
                 reject_by_annotation=True, **kwargs):
        self.kind = kind
        self.events = events
        self.event_id = event_id
        self.tmin = tmin
        self.tmax = tmax
        self.duration = duration
        self.overlap = overlap
        self.baseline = baseline
        self.picks = picks
        self.preload = preload
        self.reject_by_annotation = reject_by_annotation
        self._epoch_kwargs = kwargs
        MneTransformer.__init__(self, name='epoching', kind=kind,
                                tmin=tmin, tmax=tmax)

    def transform(self, ds):
        import mne
        if self.kind == 'fixed_length':
            logger.info('Fixed-length epoching: duration=%s overlap=%s',
                        self.duration, self.overlap)
            return mne.make_fixed_length_epochs(
                ds,
                duration=self.duration,
                overlap=self.overlap,
                preload=self.preload,
                reject_by_annotation=self.reject_by_annotation,
                **self._epoch_kwargs
            )
        logger.info('Epoching: tmin=%s tmax=%s', self.tmin, self.tmax)
        return mne.Epochs(
            ds, self.events,
            event_id=self.event_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            picks=self.picks,
            preload=self.preload,
            **self._epoch_kwargs
        )
