from typing import Callable, Dict


class Feature:
    '''Encapsulates a feature computation.

    '''

    def __init__(self, name: str, function: Callable, **kwargs: Dict) -> None:
        '''

        :param name: The human-readable name of the feature
        :param function: Function that computes the feature
        :param kwargs: Dictionary of keyword arguments that the feature accepts
        '''
        super().__init__()
        self.name = name
        self.function = function
        self.kwargs = kwargs

    def compute(self, audio, **kwargs):
        # Update with only the relevant kwargs for this function
        # self.kwargs.update(dict([(kw, kwargs[kw]) for kw in set(kwargs) & set(self.kwargs)]))
        self.kwargs.update(kwargs)
        c = self.function(y=audio, **self.kwargs)
        return c
