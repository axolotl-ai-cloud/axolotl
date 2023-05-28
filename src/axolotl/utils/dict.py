from addict import Dict


class DictDefault(Dict):
    '''
    A Dict that returns None instead of returning empty Dict for missing keys.
    '''
    def __missing__(self, key):
        return None
