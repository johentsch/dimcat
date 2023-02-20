from dataclasses import dataclass


def config_dataclass(*decorator_args, **decorator_kwargs):
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
    decorator_kwargs["frozen"] = True
    decorator_kwargs["kw_only"] = True

    def wrapper(cls):
        cls = dataclass(cls, **decorator_kwargs)
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            for fld, value in kwargs.items():
                if fld in field2enum:
                    enum_constructor = field2enum[fld]
                    kwargs[fld] = enum_constructor[kwargs[fld]]
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return wrapper(decorator_args[0]) if decorator_args else wrapper
