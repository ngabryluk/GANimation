class ModelException(Exception):
    def __init__(self, action: str):
        self.value = f'Cannot {action} a non-initialized model.'

    def __str__(self) -> str:
        return repr(self.value)


class FreezeException(Exception):
    def __init__(self, current: str, action: str):
        self.value = f'Cannot {action} a model that is already {current}.'

    def __str__(self) -> str:
        return repr(self.value)
