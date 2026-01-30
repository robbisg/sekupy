import logging
import os
logger = logging.getLogger(__name__)

def add_subdirs(path, name, sub_dirs):
    """Add subdirectories to build complete directory paths.
    
    This function creates a list of complete directory paths by combining
    a base path, subject name, and subdirectories. It validates that
    directories exist before adding them to the list.
    
    Parameters
    ----------
    path : str
        Base directory path
    name : str
        Subject or experiment name
    sub_dirs : list
        List of subdirectory names to add
        
    Returns
    -------
    list
        List of complete directory paths that exist
    """
    
    complete_path = []
    
    for d in sub_dirs:
        if d == 'none':
            d = ''
        if d[0] == '/':
            complete_path.append(d)
        
        logger.debug("%s %s %s", path, name, d)
        pathname = os.path.join(path, name, d)
        
        logger.debug(pathname)
        
        if os.path.isdir(pathname):
            complete_path.append(pathname)
    
    
    complete_path.append(path)
    complete_path.append(os.path.join(path, name))
    

    return complete_path



def build_pathnames(path, name, sub_dirs):
    """Build list of file pathnames from directories.
    
    This function creates a comprehensive list of file paths by searching
    through the base directory and specified subdirectories.
    
    Parameters
    ----------
    path : str
        Base directory path
    name : str
        Subject or experiment name
    sub_dirs : list
        List of subdirectory names to search
        
    Returns
    -------
    list
        List of file paths found in the directories
    """
            
    
    path_file_dirs = add_subdirs(path, name, sub_dirs)
    
    logger.debug(path_file_dirs)
    logger.info('Loading...')
    
    file_list = []
    # Verifying which type of task I've to classify (task or rest) 
    # and loads filename in different dirs
    for path_ in path_file_dirs:
        dir_list = [os.path.join(path_, f) for f in os.listdir(path_)]
        file_list = file_list + dir_list

    logger.debug('\n'.join(file_list))
    
    return file_list



def make_dir(path):
    """Create directory using mkdir -p command.
    
    This function wraps the Unix mkdir -p command to create
    directories recursively if they don't exist.
    
    Parameters
    ----------
    path : str
        Directory path to create
    """
    command = 'mkdir -p '+os.path.join(path)
    logger.debug(command)
    os.system(command)
    
