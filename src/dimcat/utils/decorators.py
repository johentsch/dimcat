from dataclasses import dataclass
from functools import wraps


def config_dataclass(*decorator_args, **decorator_kwargs):
    """This decorator corresponds and defaults to a @dataclass(frozen=True, kw_only=True),
    unless the keywords are overwritten. All other arguments are interpreted as (field_name = Enum) mapping.
    The resulting dataclass converts all values given as keyword arguments to the given field into the respective Enum.
    This is why it defaults to kw_only=True, making sure that there are no positional arguments that could be missed.
    """
    default_fields = (
        "init",
        "repr",
        "eq",
        "order",
        "unsafe_hash",
        "frozen",
        "match_args",
        "kw_only",
        "slots",
        "weakref_slot",
    )
    enum_fields = [fld for fld in decorator_kwargs.keys() if fld not in default_fields]
    field2enum = {}
    for fld in enum_fields:
        field2enum[fld] = decorator_kwargs.pop(fld)
    for param in ("frozen", "kw_only"):
        if param not in decorator_kwargs:
            decorator_kwargs[param] = True

    def wrapper(cls):
        cls = dataclass(cls, **decorator_kwargs)
        original_init = cls.__init__

        @wraps(cls.__init__)
        def __init__(self, *args, **kwargs):
            for fld, value in kwargs.items():
                if fld in field2enum:
                    enum_constructor = field2enum[fld]
                    kwargs[fld] = enum_constructor(kwargs[fld])
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return wrapper(decorator_args[0]) if decorator_args else wrapper
