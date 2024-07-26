import time
import logging
logger = logging.getLogger(__name__)

from sekupy.dataset.atoms import Atom
from sekupy.dataset.utils import is_datasetlike
from sekupy.dataset.base import AttrDataset


class Learner(Atom):
    """Common trainable processing object.

    A `Learner` is a `Atom` that can (maybe has to) be trained on a dataset,
    before it can perform its function.
    """

    def __init__(self, auto_train=False, force_train=False, **kwargs):
        """Initialize a Learner.

        Parameters
        ----------
        auto_train : bool
          Flag whether the learner will automatically train itself on the input
          dataset when called untrained.
        force_train : bool
          Flag whether the learner will enforce training on the input dataset
          upon every call.
        **kwargs
          All arguments are passed to the baseclass.
        """
        Atom.__init__(self, **kwargs)

        self.__is_trained = False
        self.__auto_train = auto_train
        self.__force_train = force_train

        self._training_time = None
        self._trained_targets = None
        self._trained_nsamples = None
        self._trained_dataset = None


    def train(self, ds):
        """
        The default implementation calls ``_pretrain()``, ``_train()``, and
        finally ``_posttrain()``.

        Parameters
        ----------
        ds: Dataset
          Training dataset.

        Returns
        -------
        None
        """
        got_ds = is_datasetlike(ds)

        # TODO remove first condition if all Learners get only datasets
        if got_ds and (ds.nfeatures == 0 or len(ds) == 0):
            raise Exception(
                "Cannot train learner on degenerate data %s" % ds)


        self._pretrain(ds)

        # remember the time when started training
        t0 = time.time()

        if got_ds:
            # things might have happened during pretraining
            if ds.nfeatures > 0:
                self._train(ds)

        else:
            # in this case we claim to have no idea and simply try to train
            self._train(ds)

        # store timing
        self._training_time = time.time() - t0

        # and post-proc
        self._posttrain(ds)

        # finally flag as trained
        self._set_trained()


    def untrain(self):
        """Reverts changes in the state of this node caused by previous training
        """
        # flag the learner as untrained
        # important to do that before calling the implementation in the derived
        # class, as it might decide that an object remains trained
        self._set_trained(False)
        # call subclass untrain first to allow it to access current attributes
        self._untrain()
        # TODO evaluate whether this should also reset the nodes collections, or
        # whether that should be done by a more general reset() method
        self.reset()

    def _untrain(self):
        # nothing by default
        pass

    def _pretrain(self, ds):
        """Prepare prior training.

        By default, does nothing.

        Parameters
        ----------
        ds: Dataset
          Original training dataset.

        Returns
        -------
        None
        """
        pass

    def _train(self, ds):
        # nothing by default
        pass

    def _posttrain(self, ds):
        """Finalize the training.

        By default, does nothing.

        Parameters
        ----------
        ds: Dataset
          Original training dataset.

        Returns
        -------
        None
        """
        if self._trained_targets is not None and isinstance(ds, AttrDataset):
            space = self.get_space()
            if space in ds.sa:
                self._trained_targets = ds.sa[space].unique

        self._trained_dataset = ds
        self._trained_nsamples = len(ds)

    def _set_trained(self, status=True):
        """Set the Learner's training status.

        Derived use this to set the Learner's status to trained (True) or
        untrained (False).
        """
        self.__is_trained = status

    def __call__(self, ds):
        # overwrite __call__ to perform a rigorous check whether the learner was
        # trained before use and auto-train
        if self.is_trained:
            # already trained
            if self.force_train:
                # but retraining is enforced
                self.train(ds)
        else:
            # not trained
            if self.auto_train:
                # auto training requested
                self.train(ds)
            else:
                # we always have to have trained before using a learner
                raise RuntimeError("%s needs to be trained before it can be "
                                   "used and auto training is disabled."
                                   % str(self))
        return super(Learner, self).__call__(ds)

    is_trained = property(fget=lambda x: x.__is_trained, fset=_set_trained,
                          doc="Whether the Learner is currently trained.")
    auto_train = property(fget=lambda x: x.__auto_train,
                          doc="Whether the Learner performs automatic training"
                              "when called untrained.")
    force_train = property(
        fget=lambda x: x.__force_train,
        doc="Whether the Learner enforces training upon every call.")