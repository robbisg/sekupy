import logging
from sekupy.preprocessing.mne.base import MneTransformer

logger = logging.getLogger(__name__)


class ICA(MneTransformer):
    """Fit, label, and apply ICA in a single transform step.

    Parameters
    ----------
    n_components : int
        Number of ICA components.
    method : {'fastica', 'infomax', 'picard'}
        ICA algorithm.
    label_method : {'correlation', 'iclabel'}
        Strategy for identifying bad components.
        ``'correlation'`` uses EOG/ECG channel correlation;
        ``'iclabel'`` requires ``mne-icalabel``.
    eog_ch : str or list of str, optional
        Channel name(s) for EOG-based detection.
    ecg_ch : str or list of str, optional
        Channel name(s) for ECG-based detection.
    threshold : float
        Minimum probability to exclude a component (``label_method='iclabel'``).
    random_state : int
        Random seed for reproducibility.
    max_iter : int or 'auto'
        Maximum number of iterations.
    """

    def __init__(self, n_components=20, method='fastica',
                 label_method='correlation', eog_ch=None, ecg_ch=None,
                 threshold=0.5, random_state=42, max_iter='auto', **kwargs):
        self.n_components = n_components
        self.method = method
        self.label_method = label_method
        self.eog_ch = eog_ch
        self.ecg_ch = ecg_ch
        self.threshold = threshold
        self.random_state = random_state
        self.max_iter = max_iter
        self._ica_kwargs = kwargs
        self.ica_ = None
        MneTransformer.__init__(self, name='ica',
                                n_components=n_components, method=method)

    def transform(self, ds):
        import mne.preprocessing as mnepre
        logger.info('Fitting ICA: n_components=%d method=%s',
                    self.n_components, self.method)
        self.ica_ = mnepre.ICA(
            n_components=self.n_components,
            method=self.method,
            random_state=self.random_state,
            max_iter=self.max_iter,
            **self._ica_kwargs
        )
        self.ica_.fit(ds)
        self._label(ds)
        logger.info('Excluded ICA components: %s', str(self.ica_.exclude))
        self.ica_.apply(ds)
        return ds

    def _label(self, ds):
        if self.label_method == 'iclabel':
            from mne_icalabel import label_components
            ic_labels = label_components(ds, self.ica_, method='iclabel')
            labels = ic_labels['labels']
            proba = ic_labels['y_pred_proba']
            self.ica_.exclude = [
                i for i, label in enumerate(labels)
                if label not in ('brain', 'other')
                and proba[i].max() > self.threshold
            ]
        else:
            if self.eog_ch is not None:
                eog_idx, _ = self.ica_.find_bads_eog(ds, ch_name=self.eog_ch)
                self.ica_.exclude = list(set(self.ica_.exclude + eog_idx))
            if self.ecg_ch is not None:
                ecg_idx, _ = self.ica_.find_bads_ecg(ds, ch_name=self.ecg_ch)
                self.ica_.exclude = list(set(self.ica_.exclude + ecg_idx))
