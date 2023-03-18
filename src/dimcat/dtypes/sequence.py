from __future__ import annotations

import logging
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union, overload

from dimcat.dtypes.base import PieceID, T_co, TypedSequence, WrappedDataframe
from dimcat.utils import grams, transition_matrix

# region n-grams

logger = logging.getLogger(__name__)


class ContiguousSequence(TypedSequence[T_co]):
    @overload
    def get_n_grams(self, n: Literal[2]) -> Bigrams[T_co]:
        ...

    @overload
    def get_n_grams(self, n: int) -> Ngrams[T_co]:
        ...

    def get_n_grams(self, n: int) -> Union[Ngrams[T_co], Bigrams[T_co]]:
        """
        Returns n-gram tuples of the sequence, i.e. all N-(n-1) possible direct successions of n elements.
        """
        n_grams = grams(self.values, n=n)
        if n == 2:
            return Bigrams(values=n_grams)
        else:
            return Ngrams(values=n_grams)

    def get_transition_matrix(
        self,
        n=2,
        k=None,
        smooth: int = 0,
        normalize: bool = False,
        entropy: bool = False,
        excluded_symbols: Optional[List] = None,
        distinct_only: bool = False,
        sort: bool = False,
        percent: bool = False,
        decimals: Optional[int] = None,
    ) -> WrappedDataframe:
        """Returns a transition table of n-grams, showing the frequencies with which any subsequence of length n-1
        is followed by any of the n-grams' last elements.

        Args:
            n: If ``list_of_sequences`` is passed, the number of elements per n-gram tuple. Ignored otherwise.
            k: If specified, the transition matrix will show only the top k n-grams.
            smooth: If specified, this is the minimum value in the transition matrix.
            normalize: By default, absolute counts are shown. Pass True to normalize each row.
            entropy: Pass True to add a column showing the normalized entropy for each row.
            excluded_symbols: Any n-gram containing any of these symbols will be excluded.
            distinct_only: Pass True to exclude all n-grams consisting of identical elements only.
            sort:
                By default, the columns are ordered by n-gram frequency.
                Pass True to sort them separately, i.e. each by their own frequencies.
            percent: Pass True to multiply the matrix by 100 before rounding to `decimals`
            decimals: To how many decimals you want to round the matrix, if at all.

        Returns:
            DataFrame with frequency statistics of (n-1) grams transitioning to all occurring last elements.
            The index is made up of strings corresponding to all but the last element of the n-grams,
            with the column index containing all last elements.
        """
        df = transition_matrix(
            list_of_sequences=self.values,
            n=n,
            k=k,
            smooth=smooth,
            normalize=normalize,
            entropy=entropy,
            excluded_symbols=excluded_symbols,
            distinct_only=distinct_only,
            sort=sort,
            percent=percent,
            decimals=decimals,
        )
        return WrappedDataframe(df)

    def get_changes(self) -> ContiguousSequence[T_co]:
        """Transforms values [A, A, A, B, C, C, A, C, C, C] --->  [A, B, C, A, C]"""
        prev = object()
        occurrence_list = [
            prev := v for v in self.to_series() if prev != v  # noqa: F841
        ]
        return ContiguousSequence(occurrence_list)


def to_tuple(elem: Union[T_co, Iterable[T_co]]) -> Tuple[T_co]:
    """Turns an iterable into a tuple and wraps single elements including strings in a tuple."""
    if isinstance(elem, str):
        return (elem,)
    if isinstance(elem, Iterable):
        return tuple(elem)
    return (elem,)


class Ngrams(ContiguousSequence[Tuple[T_co, ...]]):
    """N-grams know that they do not need to be converted to n-grams to compute
    a transition matrix.
    """

    def __init__(self, values: Sequence[Sequence[T_co]], converter=to_tuple, **kwargs):
        super().__init__(values, converter, **kwargs)

    def get_transition_matrix(
        self,
        n=2,
        k=None,
        smooth: int = 0,
        normalize: bool = False,
        entropy: bool = False,
        excluded_symbols: Optional[List] = None,
        distinct_only: bool = False,
        sort: bool = False,
        percent: bool = False,
        decimals: Optional[int] = None,
    ) -> WrappedDataframe:
        """Returns a transition table of n-grams, showing the frequencies with which any subsequence of length n-1
        is followed by any of the n-grams' last elements.

        Args:
            n: If ``list_of_sequences`` is passed, the number of elements per n-gram tuple. Ignored otherwise.
            k: If specified, the transition matrix will show only the top k n-grams.
            smooth: If specified, this is the minimum value in the transition matrix.
            normalize: By default, absolute counts are shown. Pass True to normalize each row.
            entropy: Pass True to add a column showing the normalized entropy for each row.
            excluded_symbols: Any n-gram containing any of these symbols will be excluded.
            distinct_only: Pass True to exclude all n-grams consisting of identical elements only.
            sort:
                By default, the columns are ordered by n-gram frequency.
                Pass True to sort them separately, i.e. each by their own frequencies.
            percent: Pass True to multiply the matrix by 100 before rounding to `decimals`
            decimals: To how many decimals you want to round the matrix, if at all.

        Returns:
            DataFrame with frequency statistics of (n-1) grams transitioning to all occurring last elements.
            The index is made up of strings corresponding to all but the last element of the n-grams,
            with the column index containing all last elements.
        """
        df = transition_matrix(
            list_of_grams=self.values,
            n=n,
            k=k,
            smooth=smooth,
            normalize=normalize,
            entropy=entropy,
            excluded_symbols=excluded_symbols,
            distinct_only=distinct_only,
            sort=sort,
            percent=percent,
            decimals=decimals,
        )
        return WrappedDataframe(df)


class Bigrams(Ngrams[Tuple[T_co, T_co]]):
    """ToDo: Need to enforce that tuples are indeed pairs."""

    pass


# endregion n-grams

# region indices


class PieceIndex(TypedSequence[PieceID], register_for=[PieceID]):
    """A sequence of :obj:`PieceID` that behaves like a set in many aspects."""

    def __init__(
        self,
        values: Sequence[Union[PieceID, Tuple[str, str]]],
        converter=PieceID._make,
        **kwargs,
    ):
        super().__init__(values=values, converter=converter, **kwargs)

    @overload
    def append(self, value: PieceID, convert: Literal[False]) -> None:
        ...

    @overload
    def append(self, value: Union[PieceID, Iterable], convert: Literal[True]) -> None:
        ...

    def append(self, value: Union[PieceID, Iterable], convert: bool = False) -> None:
        converted = self.convert(value) if convert else value
        if converted in self.values:
            logger.warning(f"Index already contains {value}.")
            return
        super().append(converted)

    def __eq__(self, other):
        return set(self.values) == set(other)

    def __hash__(self):
        return hash(tuple(set(self.values)))

    def __repr__(self):
        return f"{self.name} of length {len(self._values)}"

    def __str__(self):
        return f"{self.name} of length {len(self._values)}"


# endregion indices
