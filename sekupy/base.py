class Node(object):

    def __init__(self, name='none', **kwargs):
        self.name = name
        self._info = dict()
        
    
    def save(self, path=None):
        return