import numpy as np

def sample_generator(key, values, difference, new_targets):
    
    sample_mapper = {
                     'chunks': chunk_generator,
                     'time_coords': time_generator,
                     'events_number': event_generator,
                     'default': default_generator,
                     'targets': target_generator,
                     'stim': target_generator                    
                     }
    
    key_ = key
    if not key in sample_mapper.keys():
        key_ = 'default'

    unique = np.unique(values)
    if len(unique) == 1:
        attributes = np.array([unique[0] for _ in range(difference)])
        return np.hstack((values, attributes))
    
    return sample_mapper[key_](values, difference, new_targets)
    
    
    
def chunk_generator(values, difference, new_targets):
    
    # We hypotheses a two class problem
    y_new = new_targets[-difference:]
    y_old = new_targets[:-difference]
    less_class = y_new[-1]
    
    new_chunks = []
    for i in np.unique(values):
        
        targets = y_old[values == i]
        unique, count = np.unique(targets, return_counts=True)
        less_dict = dict(zip(unique, count))
        
        
        if len(unique) == 1:
            maj_class = 0
        else:
            maj_class = [c for k, c in less_dict.items() if k != less_class][0]
        
        count_diff = maj_class - less_dict[less_class]

        if count_diff < 0:
            class_chunk_mask = np.logical_and(values == i, y_old == less_class)
            values_ = values[class_chunk_mask]
            
            values_[count_diff:] = i+1
            values[class_chunk_mask] = values_
        else:
            new_chunks += [i for _ in range(count_diff)]
            
    return np.hstack((values, new_chunks))



def time_generator(values, difference, new_targets):
    new_values = values[:difference]
    return np.hstack((values, new_values))



def event_generator(values, difference, new_targets):
    new_values = values[:difference]
    return np.hstack((values, new_values))



def target_generator(values, difference, new_targets):
    return new_targets



def default_generator(values, difference, new_targets):   
    return np.hstack((values, values[:difference]))
      
        