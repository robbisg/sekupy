from pyitab.dataset.mappers import SliceMapper, FlattenMapper, ChainMapper
from pyitab.dataset.utils import accepts_dataset_as_samples

class FeatureSelection(SliceMapper):
    """Mapper to select a subset of features.

    Depending on the actual slicing two FeatureSelections can be merged in a
    number of ways: incremental selection (+=), union (&=) and intersection
    (|=).  Were the former assumes that two feature selections are applied
    subsequently, and the latter two assume that both slicings operate on the
    set of input features.

    Examples
    --------
    >>> from mvpa2.datasets import *
    >>> ds = Dataset([[1,2,3,4,5]])
    >>> fs0 = StaticFeatureSelection([0,1,2,3])
    >>> fs0(ds).samples
    array([[1, 2, 3, 4]])

    Merge two incremental selections: the resulting mapper performs a selection
    that is equivalent to first applying one slicing and subsequently the next
    slicing. In this scenario the slicing argument of the second mapper is
    relative to the output feature space of the first mapper.

    >>> fs1 = StaticFeatureSelection([0,2])
    >>> fs0 += fs1
    >>> fs0(ds).samples
    array([[1, 3]])
    """

    __init__doc__exclude__ = ['slicearg']

    def __init__(self, filler=0, **kwargs):
        """
        Parameters
        ----------
        filler : optional
          Value to fill empty entries upon reverse operation
        """
        # init slicearg with None
        SliceMapper.__init__(self, None, **kwargs)
        self._dshape = None
        self._oshape = None
        self.filler = filler

    def __iadd__(self, other):
        out = super(FeatureSelection, self).__iadd__(other)
        if out is self:
            # adjust our own attributes
            # if one of them was not trained, we can't say we are trained
            if self.is_trained != other.is_trained:
                self.untrain()
            elif hasattr(other, '_oshape'):
                self._oshape = other._oshape
            else:
                # we can't know now
                self._oshape = None
        elif out is NotImplemented:
            pass  # for paranoid
        else:
            raise RuntimeError("Must have not reached here")
        return out

    def _forward_data(self, data):
        """Map data from the original dataspace into featurespace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        mdata = data[:, self._slicearg]
        # store the output shape if not set yet
        if self._oshape is None:
            self._oshape = mdata.shape[1:]
        return mdata


    def _forward_dataset(self, dataset):
        # XXX this should probably not affect the source dataset, but right now
        # init_origid is not flexible enough
        if self.get_space() is not None:
            # TODO need to do a copy first!!!
            dataset.init_origids('features', attr=self.get_space())
        # invoke super class _forward_dataset, this calls, _forward_dataset
        # and this calles _forward_data in this class
        mds = super(FeatureSelection, self)._forward_dataset(dataset)
        # attribute collection needs to have a new length check
        mds.fa.set_length_check(mds.nfeatures)
        # now slice all feature attributes
        for k in mds.fa:
            mds.fa[k] = self.forward1(mds.fa[k].value)
        return mds


    def reverse1(self, data):
        # we need to reject inappropriate "single" samples to allow
        # chainmapper to properly switch to reverse() for multiple samples
        # use the fact that a single sample needs to conform to the known
        # data shape -- but may have additional appended dimensions
        if not data.shape[:len(self._oshape)] == self._oshape:
            raise ValueError("Data shape does not match training "
                             "(trained: %s; got: %s)"
                             % (self._dshape, data.shape))
        return super(FeatureSelection, self).reverse1(data)


    def _reverse_data(self, data):
        """Reverse map data from featurespace into the original dataspace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        if self._dshape is None:
            raise RuntimeError(
                "Cannot reverse-map data since the original data shape is "
                "unknown. Either set `dshape` in the constructor, or call "
                "train().")
        # this wouldn't preserve ndarray subclasses
        #mapped = np.zeros(data.shape[:1] + self._dshape,
        #                 dtype=data.dtype)
        # let's do it a little awkward but pass subclasses through
        # suggestions for improvements welcome
        mapped = data.copy() # make sure we own the array data
        # "guess" the shape of the final array, the following only supports
        # changes in the second axis -- the feature axis
        # this madness is necessary to support mapping of multi-dimensional
        # features
        mapped.resize(data.shape[:1] + self._dshape + data.shape[2:],
                      refcheck=False)
        mapped.fill(self.filler)
        mapped[:, self._slicearg] = data
        return mapped


    def _reverse_dataset(self, dataset):
        # invoke super class _reverse_dataset, this calls, _reverse_dataset
        # and this calles _reverse_data in this class
        mds = super(FeatureSelection, self)._reverse_dataset(dataset)
        # attribute collection needs to have a new length check
        mds.fa.set_length_check(mds.nfeatures)
        # now reverse all feature attributes
        for k in mds.fa:
            mds.fa[k] = self.reverse1(mds.fa[k].value)
        return mds


    @accepts_dataset_as_samples
    def _train(self, data):
        if self._dshape is None:
            # XXX what about arrays of generic objects???
            # MH: in this case the shape will be (), which is just
            # fine since feature slicing is meaningless without features
            # the only thing we can do is kill the whole samples matrix
            self._dshape = data.shape[1:]
            # we also need to know what the output shape looks like
            # otherwise we cannot reliably say what is appropriate input
            # for reverse*()
            self._oshape = data[:, self._slicearg].shape[1:]


    def _untrain(self):
        self._dshape = None
        self._oshape = None
        super(SliceMapper, self)._untrain()



class StaticFeatureSelection(FeatureSelection):
    """Feature selection by static slicing argument.
    """

    __init__doc__exclude__ = []           # slicearg is relevant again
    def __init__(self, slicearg, dshape=None, oshape=None, **kwargs):
        """
        Parameters
        ----------
        slicearg : int, list(int), array(int), array(bool)
          Any slicing argument that is compatible with numpy arrays. Depending
          on the argument the mapper will perform basic slicing or
          advanced indexing (with all consequences on speed and memory
          consumption).
        dshape : tuple
          Preseed the mappers input data shape (single sample shape).
        oshape: tuple
          Preseed the mappers output data shape (single sample shape).
        """
        FeatureSelection.__init__(self, **kwargs)
        # store it here, might be modified later
        self._dshape = self.__orig_dshape = dshape
        self._oshape = self.__orig_oshape = oshape
        # we also want to store the original slicearg to be able to reset to it
        # during training. Derived classes will override this default
        # implementation of _train()
        self.__orig_slicearg = slicearg
        self._safe_assign_slicearg(slicearg)


    @accepts_dataset_as_samples
    def _train(self, ds):
        # first thing is to reset the slicearg to the original value passed to
        # the constructor
        self._safe_assign_slicearg(self.__orig_slicearg)
        # not resetting {d,o}shape here as they will be handled upstream
        # and perform base training
        super(StaticFeatureSelection, self)._train(ds)


    def _untrain(self):
        # make trained again immediately
        self._safe_assign_slicearg(self.__orig_slicearg)
        self._dshape = self.__orig_dshape
        self._oshape = self.__orig_oshape
        super(FeatureSelection, self)._untrain()


    dshape = property(fget=lambda self: self.__orig_dshape)
    oshape = property(fget=lambda self: self.__orig_oshape)


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