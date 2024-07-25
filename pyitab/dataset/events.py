import numpy as np
from copy import copy

class Event(dict):
    """Simple class to define properties of an event.

    The class is basically a dictionary. Any properties can
    be passed as keyword arguments to the constructor, e.g.:

      >>> ev = Event(onset=12, duration=2.45)

    Conventions for keys:

    `onset`
      The onset of the event in some unit.
    `duration`
      The duration of the event in the same unit as `onset`.
    `label`
      E.g. the condition this event is part of.
    `chunk`
      Group this event is part of (if any), e.g. experimental run.
    `features`
      Any amount of additional features of the event. This might include
      things like physiological measures, stimulus intensity. Must be a mutable
      sequence (e.g. list), if present.
    """
    _MUSTHAVE = ['onset']

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : dict
          All keys to describe the Event to initialize its dict.
        """
        # store everything
        dict.__init__(self, **kwargs)

        # basic checks
        for k in Event._MUSTHAVE:
            if not k in self:
                raise ValueError("Event must have '%s' defined." % k)


    ##REF: Name was automagically refactored
    def as_descrete_time(self, dt, storeoffset=False, offsetattr='offset'):
        """Convert `onset` and `duration` information into descrete timepoints.

        Parameters
        ----------
        dt : float
          Temporal distance between two timepoints in the same unit as `onset`
          and `duration`.
        storeoffset : bool
          If True, the temporal offset between original `onset` and
          descretized onset is stored as an additional item.
        offsetattr : str
          The name of the attribute that is used to store the computed offset
          in case the `storeoffset` is enabled.

        Returns
        -------
        A copy of the original `Event` with `onset` and optionally `duration`
        replaced by their corresponding descrete timepoint. The new onset will
        correspond to the timepoint just before or exactly at the original
        onset. The new duration will be the number of timepoints covering the
        event from the computed onset timepoint till the timepoint exactly at
        the end, or just after the event.

        Note again, that the new values are expressed as #timepoint and not
        in their original unit!
        """
        dt = float(dt)
        onset = self['onset']
        out = copy(self)

        # get the timepoint just prior the onset
        out['onset'] = int(np.floor(onset / dt))

        if storeoffset:
            # compute offset
            offset = onset - (out['onset'] * dt)
            out[offsetattr] = offset

        if 'duration' in out:
            # how many timepoint cover the event (from computed onset
            # to the one timepoint just after the end of the event
            out['duration'] = int(np.ceil((onset + out['duration']) / dt) \
                                  - out['onset'])

        return out


def find_events(**kwargs):
    """Detect changes in multiple synchronous sequences.

    Multiple sequence arguments are scanned for changes in the unique value
    combination at corresponding locations. Each change in the combination is
    taken as a new event onset.  The length of an event is determined by the
    number of identical consecutive combinations.

    Parameters
    ----------
    **kwargs : sequences
      Arbitrary number of sequences that shall be scanned.

    Returns
    -------
    list
      Detected events, where each event is a dictionary with the unique
      combination of values stored under their original name. In addition, the
      dictionary also contains the ``onset`` of the event (as index in the
      sequence), as well as the ``duration`` (as number of identical
      consecutive items).

    See Also
    --------
    eventrelated_dataset : event-related segmentation of a dataset
    """
    def _build_event(onset, duration, combo):
        ev = Event(onset=onset, duration=duration, **combo)
        return ev

    events = []
    prev_onset = 0
    old_combo = None
    duration = 1
    # over all samples
    for r in range(len(list(kwargs.values())[0])):
        # current attribute combination
        combo = dict([(k, v[r]) for k, v in kwargs.items()])

        # check if things changed
        if not combo == old_combo:
            # did we ever had an event
            if old_combo is not None:
                events.append(_build_event(prev_onset, duration, old_combo))
                # reset duration for next event
                duration = 1
                # store the current samples as onset for the next event
                prev_onset = r

            # update the reference combination
            old_combo = combo
        else:
            # current event is lasting
            duration += 1

    # push the last event in the pipeline
    if old_combo is not None:
        events.append(_build_event(prev_onset, duration, old_combo))

    return events