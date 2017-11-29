def first(ordered_dict):
    """Returns first element of ordered dictionary"""
    key = next(iter(ordered_dict))
    return ordered_dict[key]


def last(ordered_dict):
    """Returns last element of ordered dictionary"""
    key = next(reversed(ordered_dict))
    return ordered_dict[key]


def nth(ordered_dict, n):
    """Returns nth element of ordered dictionary"""
    key_val = list(ordered_dict.items())[n]
    return key_val[1]
