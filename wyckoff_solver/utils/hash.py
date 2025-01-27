import hashlib
import string

import numpy as np

BASE64CHARS = string.ascii_letters + string.digits + "-_"


def uint2base64(n: int) -> str:
    """Base64-encoding non-negative integer

    Parameters
    ----------
    n : int
        Non-negative integer.

    Returns
    -------
    str
        Base64-encoded string.

    Raises
    ------
    ValueError
        n is negative

    Examples
    --------
    >>> uint2base64(0)
    'a'
    >>> list(map(uint2base64, [0, 25, 26, 51, 52, 61, 62, 63, 64]))
    ['a', 'z', 'A', 'Z', '0', '9', '-', '_', 'ab']

    """
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError(f"Non-negative integer is expected, but got {n}")
    chars = []
    while True:
        chars.append(BASE64CHARS[n % 64])
        n //= 64
        if n == 0:
            break
    return "".join(chars)


def hash_str(s, algo="sha256") -> str:
    hashfunc = getattr(hashlib, algo)
    hashdigest = hashfunc(s.encode("utf-8")).digest()
    hashint = int.from_bytes(hashdigest, "big")
    return uint2base64(hashint)


def hash_uints(uints, algo="sha256") -> str:
    """Hash a list of non-negative integers

    Parameters
    ----------
    uints : Sequence[int]
        A list of int.
    algo : str, optional
        Hash algorithom, by default "sha1".

    Returns
    -------
    str
        Hash string.

    Examples
    --------
    >>> hash_uints([1, 2, 3])
    'WklgTrrOsQ8Sp8ZR6EqPqrbEKj0m6Je9U7zu-3vE1Gl'
    """
    strcode = ",".join(uint2base64(n) for n in uints)
    return hash_str(strcode, algo)
