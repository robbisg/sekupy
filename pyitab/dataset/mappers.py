
import numpy as np
import logging
import copy
import inspect

from pyitab.dataset.learner import Learner
from pyitab.dataset.atoms import ChainAtom
from pyitab.dataset.utils import is_datasetlike, accepts_dataset_as_samples,\
    _str, _repr_attrs, is_sequence_type
from itertools import product
from functools import reduce


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

def _verified_reverse1(mapper, onesample):
    """Replacement of Mapper.reverse1 with safety net

    This function can be called instead of a direct call to a mapper's
    ``reverse1()``. It wraps a single sample into a dummy axis and calls
    ``reverse()``. Afterwards it verifies that the first axis of the
    returned array has one item only, otherwise it will issue a warning.
    This function is useful in any context where it is critical to ensure
    that reverse mapping a single sample, yields exactly one sample -- which
    isn't guaranteed due to the flexible nature of mappers.

    Parameters
    ----------
    mapper : Mapper instance
    onesample : array-like
      Single sample (in terms of the supplied mapper).

    Returns
    -------
    array
      Shape matches a single sample in terms of the mappers input space.
    """
    dummy_axis_sample = np.asanyarray(onesample)[None]
    rsample = mapper.reverse(dummy_axis_sample)
    if not len(rsample) == 1:
        logger.warning("Reverse mapping single sample yielded multiple -- can lead to unintended behavior!")
    return rsample[0]



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


class SliceMapper(Mapper):
    """Baseclass of Mapper that slice a Dataset in various ways.
    """
    def __init__(self, slicearg, **kwargs):
        """
        Parameters
        ----------
        slicearg
          Argument for slicing
        """
        Mapper.__init__(self, **kwargs)
        self._safe_assign_slicearg(slicearg)


    def _safe_assign_slicearg(self, slicearg):
        # convert int sliceargs into lists to prevent getting scalar values when
        # slicing
        if isinstance(slicearg, int):
            slicearg = [slicearg]
        self._slicearg = slicearg
        # if we got some sort of slicearg we assume that we are ready to go
        if slicearg is not None:
            self._set_trained()

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return super(SliceMapper, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['slicearg']))


    def __str__(self):
        # with slicearg it can quickly get very unreadable
        #return _str(self, str(self._slicearg))
        return _str(self)


    def _untrain(self):
        self._safe_assign_slicearg(None)
        super(SliceMapper, self)._untrain()


    def __iadd__(self, other):
        # our slicearg
        this = self._slicearg
        # if another slice mapper work on its slicearg
        if isinstance(other, SliceMapper):
            other = other._slicearg
        # catch stupid arg
        if not (isinstance(other, tuple) or isinstance(other, list) \
                or isinstance(other, np.ndarray) or isinstance(other, slice)):
            return NotImplemented
        if isinstance(this, slice):
            # we can always merge if the slicing arg can be sliced itself (i.e.
            # it is not a slice-object... unless it doesn't really slice we do
            # not want to expand slices into index lists to become mergable,
            # since that would cause cheap view-based slicing to become
            # expensive copy-based slicing
            if this == slice(None):
                # this one did nothing, just use the other and be done
                self._safe_assign_slicearg(other)
                return self
            else:
                # see comment above
                return NotImplemented
        # list or tuple are alike
        if isinstance(this, (list, tuple)):
            # simply convert it into an array and proceed from there
            this = np.asanyarray(this)
        if this.dtype.type is np.bool_:
            # simply convert it into an index array --prevents us from copying a
            # lot and allows for sliceargs such as [3,3,4,4,5,5]
            this = this.nonzero()[0]
        if this.dtype.char in np.typecodes['AllInteger']:
            self._safe_assign_slicearg(this[other])
            return self

        # if we get here we got something the isn't supported
        return NotImplemented

    slicearg = property(fget=lambda self:self._slicearg)



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
                pass
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


class FxMapper(Mapper):
    """Apply a custom transformation to (groups of) samples or features.
    """

    is_trained = True
    """Indicate that this mapper is always trained."""

    def __init__(self, axis, fx, fxargs=None, uattrs=None,
                 attrfx='merge', order='uattrs'):
        """
        Parameters
        ----------
        axis : {'samples', 'features'}
        fx : callable
        fxargs : tuple
          Passed as *args to ``fx``
        uattrs : list
          List of attribute names to consider. All possible combinations
          of unique elements of these attributes are used to determine the
          sample groups to operate on.
        attrfx : callable
          Functor that is called with each sample attribute elements matching
          the respective samples group. By default the unique value is
          determined. If the content of the attribute is not uniform for a
          samples group a unique string representation is created.
          If `None`, attributes are not altered.
        order : {'uattrs', 'occurrence', None}
          If which order groups should be merged together.  If `None` (default
          before 2.3.1), the order is imposed only by the order of
          `uattrs` as keys in the dictionary, thus can vary from run to run.
          If `'occurrence'`, groups will be ordered by the first occurrence
          of group samples in original dataset. If `'uattrs'`, groups will be
          sorted by the values of uattrs with follow-up attr having higher
          importance for ordering (e .g. `uattrs=['targets', 'chunks']` would
          order groups first by `chunks` and then by `targets` within each
          chunk).
        """
        Mapper.__init__(self)

        if not axis in ['samples', 'features']:
            raise ValueError("%s `axis` arguments can only be 'samples' or "
                             "'features' (got: '%s')." % repr(axis))
        self.__axis = axis
        self.__uattrs = uattrs
        self.__fx = fx
        if fxargs is not None:
            self.__fxargs = fxargs
        else:
            self.__fxargs = ()
        if attrfx == 'merge':
            self.__attrfx = _uniquemerge2literal
        else:
            self.__attrfx = attrfx
        assert(order in (None, 'uattrs', 'occurrence'))
        self.__order = order

    def _train(self, ds):
        # right now it needs no training, if anything is added here make sure to
        # remove is_trained class attribute
        pass

    def __smart_apply_along_axis(self, data):
        # because apply_along_axis could be very much slower than a
        # direct invocation of native functions capable of operating
        # along specific axis, let's make it smarter for those we know
        # could do that.
        fx = None
        naxis = {'samples': 0, 'features': 1}[self.__axis]
        try:
            # if first argument is 'axis' -- just proceed with a native call
            if inspect.getargs(self.__fx.__code__).args[1] == 'axis':
                fx = self.__fx
        except Exception as e:
            pass

        if fx is not None:
            mdata = fx(data, naxis, *self.__fxargs)
        else:
            # either failed to deduce signature or just didn't
            # have 'axis' second
            # apply fx along naxis for each sample/feature
            mdata = np.apply_along_axis(self.__fx, naxis, data, *self.__fxargs)
        assert(mdata.ndim in (data.ndim, data.ndim-1))
        return mdata

    def _forward_data(self, data):
        if self.__uattrs is not None:
            raise RuntimeError("%s does not support forward-mapping of plain "
                               "data when data grouping based on attributes "
                               "is requested"
                               % self.__class__.__name__)

        mdata = self.__smart_apply_along_axis(data)

        if self.__axis == 'features':
            if len(mdata.shape) == 1:
                # in case we only have a scalar per sample we need to transpose
                # it properly, to keep the length of the samples axis intact
                mdata = np.atleast_2d(mdata).T
        return np.atleast_2d(mdata)

    def _forward_dataset(self, ds):
        if self.__uattrs is None:
            mdata, sattrs = self._forward_dataset_full(ds)
        else:
            mdata, sattrs = self._forward_dataset_grouped(ds)

        samples = np.atleast_2d(mdata)

        # return early if there is no attribute treatment desired
        if self.__attrfx is None:
            out = ds.copy(deep=False)
            out.samples = samples
            return out

        # not copying the samples attributes, since they have to be modified
        # anyway
        if self.__axis == 'samples':
            out = ds.copy(deep=False, sa=[])
            col = out.sa
            incol = ds.sa
            col.set_length_check(samples.shape[0])
        else:
            out = ds.copy(deep=False, fa=[])
            col = out.fa
            incol = ds.fa
            col.set_length_check(samples.shape[1])
        # assign samples to do COW
        out.samples = samples

        for attr in sattrs:
            a = sattrs[attr]
            # TODO -- here might puke if e.g it is a list where some items
            # are empty lists... I guess just wrap in try/except and
            # do dtype=object if catch
            a = np.atleast_1d(a)
            # make sure we do not inflate the number of dimensions for no reason
            # this could happen if there was only one unique value for an
            # attribute and the default 'uniquemerge2literal' attrfx was given
            if len(a.shape) > 1 and a.shape[-1] == 1 and attr in incol \
                    and len(a.shape) > len(incol[attr].value.shape):
                a.shape = a.shape[:-1]
            col[attr] = a

        return out


    def _forward_dataset_grouped(self, ds):
        mdata = [] # list of samples array pieces
        if self.__axis == 'samples':
            col = ds.sa
            axis = 0
        elif self.__axis == 'features':
            col = ds.fa
            axis = 1
        else:
            raise RuntimeError("This should not have happened!")

        attrs = dict(zip(col.keys(), [[] for i in col]))

        # create a dictionary for all unique elements in all attribute this
        # mapper should operate on
        self.__attrcombs = dict(zip(self.__uattrs,
                                [col[attr].unique for attr in self.__uattrs]))
        # let it generate all combinations of unique elements in any attr
        order = self.order
        order_keys = []
        for comb in _orthogonal_permutations(self.__attrcombs):
            selector = reduce(np.multiply,
                                [array_whereequal(col[attr].value, value)
                                 for attr, value in comb.items()])

            # process the samples
            if axis == 0:
                samples = ds.samples[selector]
            else:
                samples = ds.samples[:, selector]

            # check if there were any samples for such a combination,
            # if not -- warning and skip the rest of the loop body
            if not len(samples):
                logger.warning('There were no samples for combination %s. It might be '
                        'a sign of a disbalanced dataset %s.' % (comb, ds))
                continue

            fxed_samples = self.__smart_apply_along_axis(samples)
            mdata.append(fxed_samples)
            if self.__attrfx is not None:
                # and now all samples attributes
                for i, attr in enumerate(col):
                    fxed_attr = self.__attrfx(col[attr].value[selector])
                    attrs[attr].append(fxed_attr)
            # possibly take care about collecting information to have groups ordered
            if order == 'uattrs':
                # reverse order as per docstring -- most of the time we have
                # used uattrs=['targets', 'chunks'] and did expect chunks being
                # groupped together.
                order_keys.append([comb[a] for a in self.__uattrs[::-1]])
            elif order == 'occurrence':
                # First index should be sufficient since we are dealing
                # with unique non-overlapping groups here (AFAIK ;) )
                order_keys.append(np.where(selector)[0][0])

        if order:
            # reorder our groups using collected "order_keys"
            # data
            order_idxs = argsort(order_keys)
            mdata = [mdata[i] for i in order_idxs]
            # and attributes
            attrs = dict((k, [v[i] for i in order_idxs])
                         for k,v in attrs.items())

        if axis == 0:
            mdata = np.vstack(mdata)
        else:
            mdata = np.vstack(np.transpose(mdata))
        return mdata, attrs


    def _forward_dataset_full(self, ds):
        # simply map the all of the data
        mdata = self._forward_data(ds.samples)

        # if the attributes should not be handled, don't handle them
        if self.__attrfx is None:
            return mdata, None

        # and now all attributes
        if self.__axis == 'samples':
            attrs = dict(zip(ds.sa.keys(),
                              [self.__attrfx(ds.sa[attr].value)
                                    for attr in ds.sa]))
        if self.__axis == 'features':
            attrs = dict(zip(ds.fa.keys(),
                              [self.__attrfx(ds.fa[attr].value)
                                    for attr in ds.fa]))
        return mdata, attrs

    axis = property(fget=lambda self:self.__axis)
    fx = property(fget=lambda self:self.__fx)
    fxargs = property(fget=lambda self:self.__fxargs)
    uattrs = property(fget=lambda self:self.__uattrs)
    attrfx = property(fget=lambda self:self.__attrfx)
    order = property(fget=lambda self:self.__order)



#
# Convenience functions to create some useful mapper with less complexity
#

def mean_sample(attrfx='merge'):
    """Returns a mapper that computes the mean sample of a dataset.

    Parameters
    ----------
    attrfx : 'merge' or callable, optional
      Callable that is used to determine the sample attributes of the computed
      mean samples. By default this will be a string representation of all
      unique value of a particular attribute in any sample group. If there is
      only a single value in a group it will be used as the new attribute value.

    Returns
    -------
    FxMapper instance.
    """
    return FxMapper('samples', np.mean, attrfx=attrfx)


def mean_group_sample(attrs, attrfx='merge', **kwargs):
    """Returns a mapper that computes the mean samples of unique sample groups.

    The sample groups are identified by the unique combination of all
    values of a set of provided sample attributes.  Order of output
    samples might differ from original and correspond to sorted order
    of corresponding `attrs`  by default.  Use `order='occurrence'` if you would
    like to maintain the order.

    Parameters
    ----------
    attrs : list
      List of sample attributes whose unique values will be used to identify the
      samples groups.
    attrfx : 'merge' or callable, optional
      Callable that is used to determine the sample attributes of the computed
      mean samples. By default this will be a string representation of all
      unique value of a particular attribute in any sample group. If there is
      only a single value in a group it will be used as the new attribute value.

    Returns
    -------
    FxMapper instance.
    """
    return FxMapper('samples', np.mean, uattrs=attrs, attrfx=attrfx, **kwargs)


def argsort(seq, reverse=False):
    """Return indices to get sequence sorted
    """
    # Based on construct from
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    # Thanks!

    # cmp was not passed through since seems to be absent in python3
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def _orthogonal_permutations(a_dict):
    """
    Takes a dictionary with lists as values and returns all permutations
    of these list elements in new dicts.

    This function is useful, when a method with several arguments
    shall be tested and all of the arguments can take several values.

    The order is not defined, therefore the elements should be
    orthogonal to each other.

    >>> for i in _orthogonal_permutations({'a': [1,2,3], 'b': [4,5]}):
    ...     print(i)
    {'a': 1, 'b': 4}
    {'a': 1, 'b': 5}
    {'a': 2, 'b': 4}
    {'a': 2, 'b': 5}
    {'a': 3, 'b': 4}
    {'a': 3, 'b': 5}
    """
    # Taken from MDP (LGPL)
    pool = dict(a_dict)
    args = []
    for func, all_args in pool.items():
        # check the size of the list in the second item of the tuple
        args_with_fun = [(func, arg) for arg in all_args]
        args.append(args_with_fun)
    for i in product(args):
        yield dict(i)
        
def array_whereequal(a, x):
    """Reliable comparison for `numpy.ndarray`

    `numpy.ndarray` (as of 1.5.0.dev) fails to compare tuples in array of
    dtype object, e.g.

    >>> import numpy as np; a=np.array([1, (0,1)], dtype=object); print(a == (0,1),  a[1] == (0,1))
    [False False] True

    This function checks if dtype is object and just does list
    comprehension in that case
    """
    if a.dtype is np.dtype('object'):
        return np.array([i==x for i in a], dtype=bool)
    else:
        return a == x

def _uniquemerge2literal(attrs):
    """Compress a sequence into its unique elements (with string merge).

    Whenever there is more then one unique element in `attrs`, these
    are converted to a string and join with a '+' character inbetween.

    Parameters
    ----------
    attrs : sequence, arbitrary

    Returns
    -------
    Non-sequence arguments are passed as is, otherwise a sequences of unique
    items is. None is returned in case of an empty sequence.
    """
    try:
        if isinstance(attrs[0], str):
            # do not try to disassemble sequences of strings
            raise TypeError
        uvalues = set(map(tuple, attrs))
        # if we were provided array of object type, most likely because
        # we had tuples or other objects, we must produce also object array
        if isinstance(attrs, np.ndarray) and attrs.dtype == 'O':
            unq = asobjarray(list(uvalues))
        else:
            unq = list(map(np.array, uvalues))
    except TypeError:
        # either no 2d-iterable...
        try:
            unq = np.unique(attrs)
        except TypeError:
            # or no iterable at all -- return the original
            return attrs

    lunq = len(unq)
    if lunq > 1:
        return ['+'.join([str(l) for l in unq])]
    elif lunq:
        return unq
    else:
        return None
    
def asobjarray(x):
    """Generates numpy.ndarray with dtype object from an iterable

    Is needed to assure object dtype, so first empty array of
    dtype=object needs to be constructed and then only items to be
    assigned.

    Parameters
    ----------
    x : list or tuple or ndarray
    """
    res = np.empty(len(x), dtype=object)
    res[:] = x
    return res