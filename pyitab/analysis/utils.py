
import logging
logger = logging.getLogger(__name__)

def get_params(param_dict, keyword):
    """[summary]
    
    Parameters
    ----------
    param_dict : [type]
        [description]
    keyword : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """

    params = dict()
    for key in param_dict.keys():
        idx = key.find(keyword)
        if idx == 0 and len(key.split("__")) > 1:
            idx += len(keyword)+2
            key_split = key[idx:]
            logger.debug(key_split)
            
            # This is for special purpose target transformer
            if key_split == "%s":
                if 'target_transformer__target' in param_dict.keys():
                    key_split = key_split %(param_dict['target_transformer__target'])
                                    
            params[key_split] = param_dict[key]

    logger.debug("%s %s" % (keyword, str(params)))
    return params