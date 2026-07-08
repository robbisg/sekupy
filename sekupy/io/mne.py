import os
import logging
from sekupy.io.configuration import read_configuration
from sekupy.io.subjects import load_subjects
from sekupy.utils.files import build_pathnames

logger = logging.getLogger(__name__)

_READERS = {
    '.fif':  'read_raw_fif',
    '.edf':  'read_raw_edf',
    '.bdf':  'read_raw_bdf',
    '.set':  'read_raw_eeglab',
    '.vhdr': 'read_raw_brainvision',
    '.cnt':  'read_raw_cnt',
    '.gdf':  'read_raw_gdf',
}


def load_raw(path, subject, img_pattern, sub_dir='', run=None, preload=True):
    """Load an MNE Raw object for a single subject.

    Parameters
    ----------
    path : str
        Base data directory.
    subject : str
        Subject identifier (subfolder under ``path``).
    img_pattern : str
        File extension used to filter files (e.g., ``'.fif'``).
    sub_dir : str
        Comma-separated subdirectory names under the subject folder.
    run : int or str, optional
        If given, only files whose name contains ``run`` are loaded.
    preload : bool
        Load data into memory immediately.

    Returns
    -------
    raw : mne.io.BaseRaw
        Concatenated Raw object (across runs if multiple files found).
    """
    import mne

    sub_dirs = sub_dir.split(',') if sub_dir else ['']
    file_list = build_pathnames(path, subject, sub_dirs)
    file_list = sorted(f for f in file_list if f.endswith(img_pattern))

    if run is not None:
        file_list = [f for f in file_list if str(run) in os.path.basename(f)]

    if not file_list:
        raise FileNotFoundError(
            "No %s file found for subject %s in %s" % (img_pattern, subject, path)
        )

    raws = []
    for fname in file_list:
        ext = os.path.splitext(fname)[-1].lower()
        reader = getattr(mne.io, _READERS.get(ext, 'read_raw'))
        logger.info('Loading %s', fname)
        raws.append(reader(fname, preload=preload))

    return mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]


class MneDataLoader(object):
    """Load MNE Raw files from a configuration file.

    Mirrors the interface of :class:`~sekupy.io.loader.DataLoader`.

    Parameters
    ----------
    configuration_file : str
        Path to the configuration (``.conf`` or ``.yaml``) file.
    task : str
        Section name in the configuration file to read.
    preload : bool
        Load raw data into memory immediately.
    **kwargs
        Override any key from the configuration file.
    """

    def __init__(self, configuration_file, task, preload=False, **kwargs):
        self._task = task
        self._preload = preload
        self._conf = read_configuration(configuration_file, task)
        self._conf.update(kwargs)

    def fetch(self, prepro=None, subject=None, run=None,
              n_subjects=None, subject_names=None):
        """Load ``(subject, mne.io.BaseRaw)`` pairs for all requested subjects.

        Parameters
        ----------
        prepro : list of MneTransformer, optional
            Preprocessing steps applied to each Raw after loading.
        subject : str, optional
            Load only this subject instead of iterating over all subjects.
        run : int or str, optional
            Load only files whose name contains this run identifier.
        n_subjects : int, optional
            Maximum number of subjects to load.
        subject_names : list of str, optional
            Subset of subjects to load by name.

        Returns
        -------
        data : list of (subject, raw)
            One tuple per subject, in load order.
        """
        pipeline = None
        if prepro is not None:
            from sekupy.preprocessing.pipelines import PreprocessingPipeline
            pipeline = PreprocessingPipeline(nodes=prepro)

        if subject is not None:
            subjects = [subject]
        else:
            subjects, _ = load_subjects(self._conf, subject_names, n_subjects)

        n = len(subjects)
        data = []
        for i, subj in enumerate(subjects):
            logger.info('Loading subject %d/%d: %s', i + 1, n, subj)
            raw = load_raw(
                self._conf['data_path'],
                subj,
                img_pattern=self._conf.get('img_pattern', '.fif'),
                sub_dir=self._conf.get('sub_dir', ''),
                run=run,
                preload=self._preload,
            )
            if pipeline is not None:
                raw = pipeline.transform(raw)
            data.append((subj, raw))

        return data

    def get_subjects(self):
        """Return the full subject list from the configuration.

        Returns
        -------
        subjects : array of str
        """
        subjects, _ = load_subjects(self._conf)
        return subjects

    def get_runs(self, subject):
        """Return available run files for a subject.

        Parameters
        ----------
        subject : str

        Returns
        -------
        runs : list of str
            Sorted list of matching file paths.
        """
        img_pattern = self._conf.get('img_pattern', '.fif')
        sub_dir = self._conf.get('sub_dir', '')
        sub_dirs = sub_dir.split(',') if sub_dir else ['']
        file_list = build_pathnames(self._conf['data_path'], subject, sub_dirs)
        return sorted(f for f in file_list if f.endswith(img_pattern))
