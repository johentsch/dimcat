def nest_level(obj, include_tuples=False):
    """Recursively calculate the depth of a nested list."""
    if obj.__class__ != list:
        if include_tuples:
            if obj.__class__ != tuple:
                return 0
        else:
            return 0
    max_level = 0
    for item in obj:
        max_level = max(max_level, nest_level(item, include_tuples=include_tuples))
    return max_level + 1


def grams(list_of_sequences, n=2):
    """Returns a list of n-gram tuples for given list. List can be nested.
    Use nesting to exclude transitions between pieces or other units.
    Uses: nest_level()

    """
    if nest_level(list_of_sequences) > 1:
        ngrams = []
        no_sublists = []
        for item in list_of_sequences:
            if isinstance(item, list):
                ngrams.extend(grams(item, n))
            else:
                no_sublists.append(item)
        if len(no_sublists) > 0:
            ngrams.extend(grams(no_sublists, n))
        return ngrams
    else:
        # if len(l) < n:
        #    print(f"{l} is too small for a {n}-gram.")
        # ngrams = [l[i:(i+n)] for i in range(len(l)-n+1)]
        ngrams = list(zip(*(list_of_sequences[i:] for i in range(n))))
        # convert to tuple of strings
        return [tuple(str(g) for g in gram) for gram in ngrams]


def get_composition_year(metadata_dict):
    start = (
        metadata_dict["composed_start"] if "composed_start" in metadata_dict else None
    )
    end = metadata_dict["composed_end"] if "composed_end" in metadata_dict else None
    if start is None and end is None:
        raise "Metadata do not include composition dates."
    if start is None:
        return end
    if end is None:
        return start
    return round((end + start) / 2, ndigits=1)


def clean_index_levels(pandas_obj):
    """Remove index levels "IDs", "corpus" and "fname", if redundant."""
    idx = pandas_obj.index
    drop = []
    if idx.nlevels > 1 and "IDs" in idx.names:
        drop.append("IDs")
    if idx.names.count("corpus") > 1:
        drop.append("corpus")
    if idx.names.count("fname") > 1:
        drop.append("fname")
    if len(drop) > 0:
        drop_levels = [idx.names.index(name) for name in drop]
        return pandas_obj.droplevel(drop_levels)
    return pandas_obj
