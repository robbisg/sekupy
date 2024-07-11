
import numpy as np
import logging

from pyitab.dataset.learner import Learner
from pyitab.dataset.atoms import ChainAtom
from pyitab.dataset.utils import is_datasetlike, accepts_dataset_as_samples, _str

logger = logging.getLogger(__name__)

def _assure_consistent_a(ds, oshape):
    """If ds.shape differs from oshape, invoke set_length_check
       for the corresponding collection
    """
    shape = ds.shape
    if oshape[0] != shape[0]:
        ds.sa.set_length_check(shape[0])
    if oshape[1] != shape[1]:
        ds.fa.set_length_check(shape[1])


class Mapper(Learner):
    """Basic mapper interface definition.

    ::

              forward
             --------->
         IN              OUT
             <--------/
               reverse

    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
          All additional arguments are passed to the baseclass.
        """
        Learner.__init__(self, **kwargs)
        # internal settings that influence what should be done to the dataset
        # attributes in the default forward() and reverse() implementations.
        # they are passed to the Dataset.copy() method
        self._sa_filter = None
        self._fa_filter = None
        self._a_filter = None

    # The following methods are abstract and merely define the intended
    # interface of a mapper and have to be implemented in derived classes. See
    # the docstrings of the respective methods for details about what they
    # should do.

    def _forward_data(self, data):
        """Forward-map some data.

        This is a private method that has to be implemented in derived
        classes.

        Parameters
        ----------
        data : anything (supported the derived class)
        """
        raise NotImplementedError


    def _reverse_data(self, data):
        """Reverse-map some data.

        This is a private method that has to be implemented in derived
        classes.

        Parameters
        ----------
        data : anything (supported the derived class)
        """
        raise NotImplementedError


    # The following methods are candidates for reimplementation in derived
    # classes, in cases where the provided default behavior is not appropriate.
    def _forward_dataset(self, dataset):
        """Forward-map a dataset.

        This is a private method that can be reimplemented in derived
        classes. The default implementation forward-maps the dataset samples
        and returns a new dataset that is a shallow copy of the input with
        the mapped samples.

        Parameters
        ----------
        dataset : Dataset-like
        """
        msamples = self._forward_data(dataset.samples)

        mds = dataset.copy(deep=False,
                           sa=self._sa_filter,
                           fa=self._fa_filter,
                           a=self._a_filter)
        mds.samples = msamples
        _assure_consistent_a(mds, dataset.shape)

        return mds


    def _reverse_dataset(self, dataset):
        """Reverse-map a dataset.

        This is a private method that can be reimplemented in derived
        classes. The default implementation reverse-maps the dataset samples
        and returns a new dataset that is a shallow copy of the input with
        the mapped samples.

        Parameters
        ----------
        dataset : Dataset-like
        """
        msamples = self._reverse_data(dataset.samples)
        mds = dataset.copy(deep=False,
                           sa=self._sa_filter,
                           fa=self._fa_filter,
                           a=self._a_filter)
        mds.samples = msamples
        _assure_consistent_a(mds, dataset.shape)

        return mds


    # The following methods provide common functionality for all mappers
    # and there should be no immediate need to reimplement them
    def forward(self, data):
        """Map data from input to output space.

        Parameters
        ----------
        data : Dataset-like, (at least 2D)-array-like
          Typically this is a `Dataset`, but it might also be a plain data
          array, or even something completely different(TM) that is supported
          by a subclass' implementation. If such an object is Dataset-like it
          is handled by a dedicated method that also transforms dataset
          attributes if necessary. If an array-like is passed, it has to be
          at least two-dimensional, with the first axis separating samples
          or observations. For single samples `forward1()` might be more
          appropriate.
        """
        if is_datasetlike(data):
            return self._forward_dataset(data)
        else:
            if hasattr(data, 'ndim') and data.ndim < 2:
                raise ValueError(
                    'Mapper.forward() only support mapping of data with '
                    'at least two dimensions, where the first axis '
                    'separates samples/observations. Consider using '
                    'Mapper.forward1() instead.')
            return self._forward_data(data)


    def forward1(self, data):
        """Wrapper method to map single samples.

        It is basically identical to `forward()`, but also accepts
        one-dimensional arguments. The map whole dataset this method cannot
        be used. but `forward()` handles them.
        """
        if isinstance(data, np.ndarray):
            data = data[np.newaxis]
        else:
            data = np.array([data])

        return self.forward(data)[0]



    def reverse(self, data):
        """Reverse-map data from output back into input space.

        Parameters
        ----------
        data : Dataset-like, anything
          Typically this is a `Dataset`, but it might also be a plain data
          array, or even something completely different(TM) that is supported
          by a subclass' implementation. If such an object is Dataset-like it
          is handled by a dedicated method that also transforms dataset
          attributes if necessary.
        """
        if is_datasetlike(data):
            return self._reverse_dataset(data)
        else:
            return self._reverse_data(data)


    def reverse1(self, data):
        """Wrapper method to map single samples.

        It is basically identical to `reverse()`, but accepts one-dimensional
        arguments. To map whole dataset this method cannot be used. but
        `reverse()` handles them.
        """
        if isinstance(data, np.ndarray):
            data = data[np.newaxis]
        else:
            data = np.array([data])
        mapped = self.reverse(data)[0]
        return mapped

    def _call(self, ds):
        return self.forward(ds)


class FlattenMapper(Mapper):
    """Reshaping mapper that flattens multidimensional arrays into 1D vectors.

    This mapper performs relatively cheap reshaping of arrays from ND into 1D
    and back upon reverse-mapping. The mapper has to be trained with a data
    array or dataset that has the first axis as the samples-separating
    dimension. Mapper training will set the particular multidimensional shape
    the mapper is transforming into 1D vector samples. The setting remains in
    place until the mapper is retrained.

    Notes
    -----
    At present this mapper is only designed (and tested) to work with C-ordered
    arrays.
    """
    def __init__(self, shape=None, maxdims=None, **kwargs):
        """
        Parameters
        ----------
        shape : tuple
          The shape of a single sample. If this argument is given the mapper
          is going to be fully configured and no training is necessary anymore.
        maxdims : int or None
          The maximum number of dimensions to flatten (starting with the first).
          If None, all axes will be flattened.
        """
        # by default auto train
        kwargs['auto_train'] = kwargs.get('auto_train', True)
        Mapper.__init__(self, **kwargs)
        self.__origshape = None         # pylint pacifier
        self.__maxdims = maxdims
        if shape is not None:
            self._train_with_shape(shape)

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return super(FlattenMapper, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['shape', 'maxdims']))

    def __str__(self):
        return _str(self)


    @accepts_dataset_as_samples
    def _train(self, samples):
        """Train the mapper.

        Parameters
        ----------
        samples : array-like
          The first axis has to represent the samples-separating dimension. In
          case of a 1D-array each element is considered to be an individual
          element and *not* the whole array as a single sample!
        """
        self._train_with_shape(samples.shape[1:])

    def _train_with_shape(self, shape):
        """Configure the mapper with a particular sample shape."""
        # infer the sample shape from the data under the assumption that the
        # first axis is the samples-separating dimension
        self.__origshape = shape
        # flag the mapper as trained
        self._set_trained()


    def _forward_data(self, data):
        # this method always gets data where the first axis is the samples axis!
        # local binding
        nsamples = data.shape[0]
        sshape = data.shape[1:]
        oshape = self.__origshape

        if oshape is None:
            raise RuntimeError("FlattenMapper needs to be trained before it "
                               "can be used.")
        # at least the first feature axis has to match match
        if oshape[0] != sshape[0]:
            raise ValueError("FlattenMapper has not been trained for data "
                             "shape '%s' (known only '%s')."
                             % (str(sshape), str(oshape)))

        if self.__maxdims is not None:
            maxdim = min(len(oshape), self.__maxdims)
        else:
            maxdim = len(oshape)
        # flatten the pieces the mapper knows about and preserve the rest
        return data.reshape((nsamples, -1) + sshape[maxdim:])


    def _forward_dataset(self, dataset):
        # invoke super class _forward_dataset, this calls, _forward_dataset
        # and this calls _forward_data in this class
        mds = super(FlattenMapper, self)._forward_dataset(dataset)
        # attribute collection needs to have a new length check
        mds.fa.set_length_check(mds.nfeatures)
        # we need to duplicate all existing feature attribute, as each original
        # feature is now spread across the new feature axis
        # take all "additional" axes after the actual feature axis and count
        # elements a sample -- if not axis exists this will be 1
        for k in dataset.fa:

            attr = dataset.fa[k].value
            # the maximmum number of axis to flatten in the attr
            if self.__maxdims is not None:
                maxdim = min(len(self.__origshape), self.__maxdims)
            else:
                maxdim = len(self.__origshape)
            multiplier = mds.nfeatures \
                    / np.prod(attr.shape[:maxdim])

            # broadcast as many times as necessary to get 'matching dimensions'
            bced = np.repeat(attr, multiplier, axis=0)
            # now reshape as many dimensions as the mapper knows about
            mds.fa[k] = bced.reshape((-1,) + bced.shape[maxdim:])

        # if there is no inspace return immediately
        if self.get_space() is None:
            return mds
        # otherwise create the coordinates as feature attributes
        else:
            mds.fa[self.get_space()] = \
                list(np.ndindex(dataset.samples[0].shape))
            return mds


    def _reverse_data(self, data):
        # this method always gets data where the first axis is the samples axis!
        # local binding
        nsamples = data.shape[0]
        sshape = data.shape[1:]
        oshape = self.__origshape
        return data.reshape((nsamples,) + oshape + sshape[1:])


    def _reverse_dataset(self, dataset):
        # invoke super class _reverse_dataset, this calls, _reverse_dataset
        # and this calles _reverse_data in this class
        mds = super(FlattenMapper, self)._reverse_dataset(dataset)
        # attribute collection needs to have a new length check
        mds.fa.set_length_check(mds.nfeatures)
        # now unflatten all feature attributes
        inspace = self.get_space()
        for k in mds.fa:
            # reverse map all attributes, but not the inspace indices, since the
            # did not come through this mapper and make not sense in inspace
            if k != inspace:
                mds.fa[k] = _verified_reverse1(self, mds.fa[k].value)
        # wipe out the inspace attribute -- needs to be done after the loop to
        # not change the size of the dict
        if inspace and inspace in mds.fa:
            del mds.fa[inspace]
        return mds

    shape = property(fget=lambda self: self.__origshape)
    maxdims = property(fget=lambda self: self.__maxdims)
    


class ChainMapper(ChainAtom):
    """Class that amends ChainNode with a mapper-like interface.

    ChainMapper supports sequential training of a mapper chain, as well as
    reverse-mapping and mapping of single samples.
    """
    def forward(self, ds):
        return self(ds)


    def forward1(self, data):
        """Forward data or datasets through the chain.

        See `Mapper` for more information.
        """
        mp = data
        for m in self:
            mp = m.forward1(mp)
        return mp


    def reverse(self, data):
        """Reverse-maps data or datasets through the chain (backwards).

        See `Mapper` for more information.
        """
        mp = data
        for m in reversed(self):
            # we ignore mapper that do not have reverse mapping implemented
            # (e.g. detrending). That might cause problems if ignoring the
            # mapper make the data incompatible input for the next mapper in
            # the chain. If that pops up, we have to think about a proper
            # solution.
            try:
                mp = m.reverse(mp)
            except NotImplementedError:
                if __debug__:
                    debug('MAP', "Ignoring %s on reverse mapping." % m)
        return mp


    def reverse1(self, data):
        """Reverse-maps data or datasets through the chain (backwards).

        See `Mapper` for more information.
        """
        mp = data
        for i, m in enumerate(reversed(self)):
            # we ignore mapper that do not have reverse mapping implemented
            # (e.g. detrending). That might cause problems if ignoring the
            # mapper make the data incompatible input for the next mapper in
            # the chain. If that pops up, we have to think about a proper
            # solution.
            try:
                mp = m.reverse1(mp)
            except NotImplementedError:
                pass
            except ValueError:
                mp = self[:-1 * i].reverse(mp)
                return mp
        return mp


    def train(self, dataset):
        """Train the mapper chain sequentially.

        The training dataset is used to train the first mapper. Afterwards it is
        forward-mapped by this (now trained) mapper and the transformed dataset
        and then used to train the next mapper. This procedure is done till all
        mappers are trained.

        Parameters
        ----------
        dataset: `Dataset`
        """
        nmappers = len(self) - 1
        tdata = dataset
        for i, mapper in enumerate(self):
            mapper.train(tdata)
            # forward through all but the last mapper
            if i < nmappers:
                tdata = mapper.forward(tdata)


    def untrain(self):
        """Untrain all embedded mappers."""
        for m in self:
            m.untrain()


    def __str__(self):
        return super(ChainMapper, self).__str__().replace('Mapper', '')



def mask_mapper(mask=None, shape=None, space=None):
    """Factory method to create a chain of Flatten+StaticFeatureSelection Mappers

    Parameters
    ----------
    mask : None or array
      an array in the original dataspace and its nonzero elements are
      used to define the features included in the dataset. Alternatively,
      the `shape` argument can be used to define the array dimensions.
    shape : None or tuple
      The shape of the array to be mapped. If `shape` is provided instead
      of `mask`, a full mask (all True) of the desired shape is
      constructed. If `shape` is specified in addition to `mask`, the
      provided mask is extended to have the same number of dimensions.
    inspace
      Provided to `FlattenMapper`
    """
    if mask is None:
        if shape is None:
            raise ValueError("Either `shape` or `mask` have to be specified.")
        else:
            # make full dataspace mask if nothing else is provided
            mask = np.ones(shape, dtype='bool')
    else:
        if shape is not None:
            # expand mask to span all dimensions but first one
            # necessary e.g. if only one slice from timeseries of volumes is
            # requested.
            mask = np.array(mask, copy=False, subok=True, ndmin=len(shape))
            # check for compatibility
            if not shape == mask.shape:
                raise ValueError(\
                    "The mask dataspace shape %s is not " \
                    "compatible with the provided shape %s." \
                    % (mask.shape, shape))

    fm = FlattenMapper(shape=mask.shape, space=space)
    flatmask = fm.forward1(mask)
    mapper = ChainMapper([fm,
                          StaticFeatureSelection(
                              flatmask,
                              dshape=flatmask.shape,
                              oshape=(len(flatmask.nonzero()[0]),))])
    return mapper