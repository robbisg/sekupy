import os
import logging
import pandas as pd
logger = logging.getLogger(__name__)


def find_directory(path, **kwargs):
    """[summary]

    Parameters
    ----------
    path : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    dir_analysis = os.listdir(path)
    dir_analysis.sort()
    
    filtered_dirs = filter_bids(dir_analysis, **kwargs)

    return filtered_dirs


def filter_bids(filelist, **filters):
    # TODO: Test!
    bidslist = [get_dictionary(f) for f in filelist]

    logger.debug(bidslist)
    logger.debug(filters)

    filtered = []
    for key, value in filters.items():
        for dictionary in bidslist:
            if key in dictionary.keys():
                #value = value.replace("_", "+")
                if dictionary[key] in value:
                    filtered.append(dictionary)

    logger.debug(filtered)

    return filtered

def filter_files(filelist, **filters):
    """Filters BIDS-style files. 
    The key to filter is specified in ```filters```:
    only those contained in filters are checked all missing 
    are included in the final filelist.

    Parameters
    ----------
    filelist : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    bidslist = [get_dictionary(f) for f in filelist]

    mask = [True for _ in bidslist]

    for key, value in filters.items():
        for i, dictionary in enumerate(bidslist):
            if key in dictionary.keys() and mask[i]:
                if dictionary[key] not in value:
                    mask[i] = False

    bidslist = [b for i, b in enumerate(bidslist) if mask[i]]
    logger.debug(bidslist)

    return bidslist


# TODO: Generalize for dirs and files
def get_dictionary(filename):
    """[summary]

    Parameters
    ----------
    filename : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    dictionary = dict()

    parts = filename.split("_")

    index = [i for i, f in enumerate(parts) if f.find("-") == -1]

    # If some parts haven't the pattern key-value then drop it.
    # TODO: Send an Exception?
    if len(index) == len(parts):
        return dictionary

    new_parts = []
    for i in index:
        part = parts[i]
        logger.debug(part)
        if i == len(parts) - 1:
            pp = part.split(".")

            if len(pp) == 3:
                trailing = pp[0]
                ext = "%s.%s" % (pp[1], pp[2])
            elif len(pp) == 2:
                trailing, ext = pp
            else:
                trailing = '+'.join(pp[:-2])
                ext = '.'.join(pp[-2:])

            new_parts.append("filetype-%s" % (trailing))
            new_parts.append("extension-%s" % (ext))

        if i == 0:
            new_parts.append("subjecttype-%s" %(part))

    parts += new_parts
    logger.debug(parts)

    for part in parts:
        try:
            key, value = part.split("-")
        except Exception as err:
            continue

        dictionary[key] = value
    
    dictionary['filename'] = filename

    part = parts[0]
    path_parts = part.split('/')
    if 'pipeline' not in dictionary.keys():
        if 'derivatives' in path_parts:
            pipeline = path_parts[path_parts.index('derivatives')+1]
        else:
            pipeline = 'raw'

        dictionary['pipeline'] = pipeline

    return dictionary


def write_participants_tsv(path, dataframe=None):

    if dataframe is not None and isinstance(dataframe, pd.DataFrame):
        dataframe.to_csv(os.path.join(path, "participants.tsv"), 
                         sep="\t", index=False)

    
    return


def dict_to_fname(dictionary):
    return "_".join(["%s-%s" % (k, v) for k, v in dictionary.items()])
