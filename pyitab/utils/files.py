import logging
import os
logger = logging.getLogger(__name__)

def add_subdirs(path, name, sub_dirs):
    
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
    """ Make dir unix command wrapper """
    #os.mkdir(os.path.join(path))
    command = 'mkdir -p '+os.path.join(path)
    logger.debug(command)
    os.system(command)
    
