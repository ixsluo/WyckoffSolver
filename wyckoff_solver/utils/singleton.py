import threading
from abc import abstractmethod, ABCMeta
from inspect import signature, BoundArguments
from queue import Queue
from typing import Any


class ParamSingletonFactory:
    """Factory to create metaclass for Singleton by parameters

    This class is a singleton itself to store all metaclasses. The created metaclass
    is to create other singletons distinguished by parameters.

    The metaclass is created by ``create_metaclass(name, cache_size)`` method.

    Warning
    -------
    All subclass inherits the created metaclass MUST override the classmethod
    ``encode_arguments``, which returns a hashable object using the
    ``inspect.BoundArguments`` instance of __init__ to distinguish each singleton.

    Examples
    --------
    >>> MetaA = ParamSingletonFactory.create_metaclass("A", cache_size=2)
    >>> class A(metaclass=MetaA):
            def __init__(self, a):
                pass

            @classmethod
            def encode_arguments(cls, bound_args):
                return str(bound_args.arguments['a'])
    >>> A(1) is A(1)
    True
    >>> MetaB = ParamSingletonFactory.create_metaclass("B")
    >>> class B(metaclass=MetaB):
            def __init__(self, b, **kwargs):
                pass

            @classmethod
            def encode_arguments(cls, bound_args):
                return (str(bound_arguments['b']), frozenset(bound_args.kwargs.items()))
    >>> A(1) is B(1)
    False

    """
    _instance = None
    _lock = threading.Lock()
    _metaclass_cache = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if cls._instance is None:
                        cls._instance = super().__call__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def create_metaclass(cls, name: str, cache_size=0):
        if name not in cls._metaclass_cache:

            def call(cls, *args, **kwargs):
                sig = signature(cls.__init__)
                bound_args = sig.bind_partial(cls, *args, **kwargs)
                bound_args.apply_defaults()
                if not hasattr(cls, "_instance_queue"):
                    cls._instance_queue = Queue(cache_size)
                if not hasattr(cls, "_instance"):
                    cls._instance = {}
                if not hasattr(cls, "_lock"):
                    cls._lock = threading.Lock()

                cls_code = cls.encode_arguments(bound_args)
                key = (cls, cls_code)
                try:
                    hash(key)
                except TypeError:
                    raise TypeError("The 'encode_arguments' or other kwargs is unhashable.")
                if key not in cls._instance:
                    with cls._lock:
                        if key not in cls._instance:
                            instance = super(type(cls), cls).__call__(*args, **kwargs)
                            instance._cls_code = cls_code
                            print(instance._cls_code)
                            if cls._instance_queue.full():
                                drop_key = cls._instance_queue.get_nowait()
                                cls._instance.pop(drop_key)
                            cls._instance_queue.put_nowait(key)
                            cls._instance[key] = instance
                            return instance
                return cls._instance[key]

            def encode_arguments(cls, bound_args: BoundArguments):
                raise NotImplemented

            new_meta = type(
                f"ParamSingleton_{name}",
                (type,),
                {
                    '__call__': call,
                    'encode_arguments': classmethod(abstractmethod(encode_arguments)),
                }
            )
            cls._metaclass_cache[name] = new_meta
        return cls._metaclass_cache[name]
