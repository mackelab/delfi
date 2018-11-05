import abc

from functools import partial


class ABCMetaDoc(type, metaclass=abc.ABCMeta):
    """
    Source: http://code.activestate.com/recipes/578587-inherit-method-docstrings-without-breaking-decorat/
    """
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        classdict = super().__prepare__(name, bases, *kwds)

        # Inject decorators into class namespace
        classdict['copy_ancestor_docstring'] = partial(
            _copy_ancestor_docstring, mro(*bases))

        return classdict

    def __new__(cls, name, bases, classdict):

        # Decorator may not exist in class dict if the class (metaclass
        # instance) was constructed with an explicit call to `type`.
        # (cf http://bugs.python.org/issue18334)
        if 'copy_ancestor_docstring' in classdict:

            # Make sure that class definition hasn't messed with decorators
            copy_impl = getattr(
                classdict['copy_ancestor_docstring'], 'func', None)
            if copy_impl is not _copy_ancestor_docstring:
                raise RuntimeError(
                    'No copy_ancestor_docstring attribute may be created '
                    'in classes using the InheritableDocstrings metaclass')

            # Delete decorators from class namespace
            del classdict['copy_ancestor_docstring']

        return super().__new__(cls, name, bases, classdict)


def _copy_ancestor_docstring(mro, fn):
    """Decorator to set docstring for *fn* from *mro*

    Source: http://code.activestate.com/recipes/578587-inherit-method-docstrings-without-breaking-decorat/
    """
    if fn.__doc__ is not None:
        raise RuntimeError('Function already has docstring')

    # Search for docstring in superclass
    for cls in mro:
        super_fn = getattr(cls, fn.__name__, None)
        if super_fn is None:
            continue
        fn.__doc__ = super_fn.__doc__
        break
    else:
        raise RuntimeError("Can't inherit docstring for %s: method does not "
                           "exist in superclass" % fn.__name__)

    return fn


def mro(*bases):
    """Calculate the Method Resolution Order of bases using the C3 algorithm.

    Suppose you intended creating a class K with the given base classes. This
    function returns the MRO which K would have, *excluding* K itself (since
    it doesn't yet exist), as if you had actually created the class.

    Another way of looking at this, if you pass a single class K, this will
    return the linearization of K (the MRO of K, *including* itself).

    Source: http://code.activestate.com/recipes/577748-calculate-the-mro-of-a-class/
    """
    seqs = [list(C.__mro__) for C in bases] + [list(bases)]
    res = []
    while True:
        non_empty = list(filter(None, seqs))
        if not non_empty:
            # Nothing left to process, we're done.
            return tuple(res)
        for seq in non_empty:  # Find merge candidates among seq heads.
            candidate = seq[0]
            not_head = [s for s in non_empty if candidate in s[1:]]
            if not_head:
                # Reject the candidate.
                candidate = None
            else:
                break
        if not candidate:
            raise TypeError("inconsistent hierarchy, no C3 MRO is possible")
        res.append(candidate)
        for seq in non_empty:
            # Remove candidate.
            if seq[0] == candidate:
                del seq[0]
