class BaseAlgorithmException(Exception):
    def __init__(self, error: str, *args, **kwargs):
        super().__init__(f"{self.__class__.__name__}: {error}", *args, **kwargs)
