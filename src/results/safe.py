from __future__ import annotations

import abc
import typing as t

from typing_extensions import Final, TypeAlias

__all__ = [
    "Err",
    "Nothing",
    "Ok",
    "Option",
    "Result",
    "Some",
    "NOTHING",
]

T = t.TypeVar("T")  # Success type
U = t.TypeVar("U")
E = t.TypeVar("E", covariant=True)  # Error type
F = t.TypeVar("F")
R = t.TypeVar("R")
OT = t.TypeVar("OT", bound="Option[t.Any]")
RT = t.TypeVar("RT", bound="ResultABC[t.Any, t.Any]")


class SingletonABC(abc.ABCMeta):
    _instances: dict[t.Type[SingletonABC], SingletonABC] = {}
    __slots__ = ()

    def __call__(cls, *args: t.Any, **kwargs: t.Any):  # type: ignore[no-untyped-def]
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonABC, cls).__call__(*args, **kwargs)  # type: ignore[index]
        return cls._instances[cls]  # type: ignore[index]


class OptionABC(t.Generic[T], metaclass=abc.ABCMeta):
    __slots__ = ()

    def __init_subclass__(cls, *args: t.Any, **kwargs: t.Any):
        super().__init_subclass__(*args, **kwargs)
        abc_members = [
            member for member in dir(OptionABC) if not member.startswith("_")
        ]
        cls_members = [
            member
            for member in dir(cls)
            if not member.startswith("_")
            and not getattr(getattr(cls, member), "__isabstractmethod__", False)
        ]
        if abc_members != cls_members:
            raise TypeError(
                f"Invalid Option interface. Unknown methods: {set(cls_members).difference(abc_members) or '{}'}. Missing methods: {set(abc_members).difference(cls_members)}"
            )

    @property
    @abc.abstractmethod
    def value(self) -> T:
        """Get option value.

        Returns
            The contained `Some` value or raise an error if option is `Nothing`.

        Raises:
            `NothingOptionError`: when option is `Nothing()`.

        Examples:

        - Access some value

        >>> Some(1).value
        1

        - Fail to access value from nothing

        >>> Nothing().value
        Traceback (most recent call last):
            ...
        results.safe.NothingOptionError: Nothing objects do not have value
        """

    @abc.abstractmethod
    def __repr__(self) -> str:
        """String representation."""

    @abc.abstractmethod
    def __eq__(self, other: t.Any) -> bool:
        """Support for `==` operator.

        Examples:

        >>> Some(1) == Some(1)
        True
        >>> Some(1) == Some(0)
        False
        >>> Some(1) == 1
        False
        >>> Nothing() == Nothing()
        True
        >>> Some(1) == Nothing()
        False
        """

    @abc.abstractmethod
    def __ne__(self, other: t.Any) -> bool:
        """Support for `!=` operator.

        Examples:

        >>> Some(1) != Some(1)
        False
        >>> Some(1) != Some(0)
        True
        >>> Some(1) != 1
        True
        >>> Nothing() != Nothing()
        False
        >>> Some(1) != Nothing()
        True
        >>> Nothing() != Some(1)
        True
        """

    @abc.abstractmethod
    def __hash__(self) -> int:
        """Option hash value.

        Providing a `__hash__` method for `Option` classes is useful
        because it allows using structures such as `set` or tools
        such as `functools.lru_cache`.

        Examples:

        - Reduce a list of 3 duplicate `Some` options into a set of 1 `Some` option

        >>> set([Some(1), Some(1), Some(1)])
        {Some(1)}

        - Reduce a list of duplicates `Nothing()` objects

        >>> set([Nothing(), Nothing(), Nothing()])
        {Nothing()}
        """

    @abc.abstractmethod
    def __bool__(self) -> bool:
        """Boolean operator."""

    @abc.abstractmethod
    def __iter__(self) -> t.Iterator[T]:
        """Support iteration over option value.

        The returned iterator yield a single value in case of `Some` option
        else does not yield value.

        Examples:

        - `Some` option behaves like an iterator yielding the contained value

        >>> list(Some(1))
        [1]

        - `Nothing()` option behaves like an empty iterator

        >>> list(Nothing())
        []
        """

    @abc.abstractmethod
    def __next__(self) -> T:
        """Return result value in case of Some else raise a StopIteration."""

    @abc.abstractmethod
    def is_some(self) -> t.Literal[True, False]:
        """Check if option is `Some`.

        Returns:
            `True` if the option is `Some`, else `False`.

        Examples:

        >>> Some(1).is_some()
        True
        >>> Some(False).is_some()
        True
        >>> Some(None).is_some()
        True
        >>> Nothing().is_some()
        False
        """

    @abc.abstractmethod
    def is_some_and(self, predicate: t.Callable[[T], bool]) -> bool:
        """Check if option is `Some` and contained value matches a predicate.

        Returns:
            `True` if the option is `Some` and contained value matches predicate, else `False`.

        Examples:

        - Option is `Some` and predicate matches

        >>> Some(True).is_some_and(lambda v: v is True)
        True

        - Option is `Some` and predicates does not match

        >>> Some(False).is_some_and(lambda v: v is True)
        False

        - Predicate is not used when option is `Nothing`

        >>> Nothing().is_some_and(...)
        False
        """

    @abc.abstractmethod
    def is_nothing(self) -> t.Literal[True, False]:
        """Check if option is `Nothing`.

        Returns:
            `True` if the option is `Nothing`, else `False`.

        Examples:

        - Option is `Some`

        >>> Some(True).is_nothing()
        False

        - Option is `Nothing`

        >>> Nothing().is_nothing()
        True
        """

    @abc.abstractmethod
    def unwrap_nothing(self, msg: str = "") -> None:
        """Returns the None if option is `Nothing()`or raise a `IsSomeError` with message provided by `msg` argument.

        Returns:
            None

        Raises:
            `SomeOptionError`: when option is `Some`.

        Examples:

        - An exception is raised

        >>> Some(1).unwrap_nothing("failure")
        Traceback (most recent call last):
            ...
        results.safe.SomeOptionError: failure

        - Get nothing

        >>> Nothing().unwrap_nothing("failure")
        """

    @abc.abstractmethod
    def unwrap(self, msg: str = "") -> T:
        """Returns the contained `Some` value or raise a `NothingOptionError` with message provided by `msg` argument.

        Returns:
            The contained value.

        Raises:
            `NothingOptionError`: when option is `Nothing()`.

        Examples:

        - Get value

        >>> Some(1).unwrap("failure")
        1

        - Fail to get value

        >>> Nothing().unwrap("failure")
        Traceback (most recent call last):
            ...
        results.safe.NothingOptionError: failure
        """

    @abc.abstractmethod
    def unwrap_or(self, default: T) -> T:
        """Returns the contained `Some` value or a default value.

        Examples:

        - Default value is ignored when option is `Some`

        >>> Some(1).unwrap_or(2)
        1

        - Default value is always returned when option is `Nothing`

        >>> Nothing().unwrap_or(2)
        2
        """

    @abc.abstractmethod
    def unwrap_or_else(self, func: t.Callable[[], T]) -> T:
        """Returns the contained `Some` value or a default value obtained by calling givne function.

        Examples:

        - Function is ignored when option is `Some`

        >>> Some(1).unwrap_or_else(lambda: 2)
        1

        - Function return value is always returned when option is `Nothing`

        >>> Nothing().unwrap_or_else(lambda: 2)
        2
        """

    @abc.abstractmethod
    def map(self, func: t.Callable[[T], U]) -> Option[U]:
        """Transform value if option is `Some`.

        Returns:
            `Some[U]` if option is `Some` else `Nothing`.

        Examples:

        >>> Some(1).map(lambda x: x + 2)
        Some(3)

        >>> Nothing().map(lambda x: x + 2)
        Nothing()
        """

    @abc.abstractmethod
    def inspect(self, func: t.Callable[[T], t.Any]) -> Option[T]:
        """Call function on contained value if option is `Some` and returns option.

        Returns:
            self

        Examples:

        >>> Some(mutable := {"x": 0}).inspect(lambda mutable: None)
        Some({'x': 0})

        - Function is ignored when option is `Nothing`

        >>> Nothing().inspect(...)
        Nothing()
        """

    @abc.abstractmethod
    def ok_or(self, err: F) -> Result[T, F]:
        """Transforms the `Option[T]` into a `Result[T, E]`, mapping `Some(v)` to `Ok(v)` and `Nothing` to `Err(err)`.

        Examples:

        >>> Some(1).ok_or("failure")
        Ok(1)
        >>> Nothing().ok_or("failure")
        Err('failure')
        """

    @abc.abstractmethod
    def ok_or_else(self, err: t.Callable[[], F]) -> Result[T, F]:
        """Transforms the `Option[T]` into a `Result[T, E]`, mapping `Some(v)` to `Ok(v)` and `Nothing` to `Err(err())`.

        Examples:

        >>> Some(1).ok_or_else(lambda: "failure")
        Ok(1)
        >>> Nothing().ok_or_else(lambda: "failure")
        Err('failure')
        """

    @abc.abstractmethod
    def and_option(self, other: Option[U]) -> Option[U]:
        """Returns `Nothing` if the option is `Nothing` or other is `Nothing`, otherwise returns other

        Examples:

        - Other is always returned when option is `Some`

        >>> Some(1).and_option(Some(2))
        Some(2)
        >>> Some(1).and_option(Nothing())
        Nothing()

        - `Nothing()` is always returned when option is `Nothing`

        >>> Nothing().and_option(Some(1))
        Nothing()

        >>> Nothing().and_option(Nothing())
        Nothing()
        """

    @abc.abstractmethod
    def or_option(self, other: Option[T]) -> Option[T]:
        """Returns the option if it contains a value, otherwise returns other.

        Examples:

        - Other is always returned when option is `Nothing`

        >>> Nothing().or_option(Some(2))
        Some(2)
        >>> Nothing().or_option(Nothing())
        Nothing()

        - Option is returned when other is `Nothing()`

        >>> Some(1).or_option(Nothing())
        Some(1)

        - Option is returned when boths option and other are `Some`

        >>> Some(1).or_option(Some(2))
        Some(1)
        """

    @abc.abstractmethod
    def xor_option(self, other: Option[T]) -> Option[T]:
        """Returns `Some` if exactly one of self, other is `Some`, otherwise returns `Nothing`.

        Examples:

        >>> Some(1).xor_option(Some(2))
        Nothing()

        >>> Some(1).xor_option(Nothing())
        Some(1)

        >>> Nothing().xor_option(Some(1))
        Some(1)

        >>> Nothing().xor_option(Nothing())
        Nothing()
        """

    @abc.abstractmethod
    def bind(self, func: t.Callable[[T], Option[U]]) -> Option[U]:
        """Transform `Some` option.

        Function is ignored when option is `Nothing()`.

        Examples:

        >>> Some(1).bind(lambda x: Some(x+1))
        Some(2)

        >>> Some(1).bind(lambda x: Nothing())
        Nothing()

        >>> Nothing().bind(lambda x: Some(x+1))
        Nothing()
        """

    @abc.abstractmethod
    def or_else(self, func: t.Callable[[], Option[T]]) -> Option[T]:
        """Returns the option if it contains a value, otherwise calls provided function and return option.

        Examples:

        - Other is always returned when option is `Nothing()`

        >>> Nothing().or_else(lambda: Some(2))
        Some(2)
        >>> Nothing().or_else(lambda: Nothing())
        Nothing()

        - Option is returned when other is `Nothing()`

        >>> Some(1).or_else(lambda: Nothing())
        Some(1)

        - Option is returned when boths option and other are `Some`

        >>> Some(1).or_else(lambda: Some(2))
        Some(1)
        """

    @abc.abstractmethod
    def filter(self, predicate: t.Callable[[T], bool]) -> Option[T]:
        """Return `Some` only if option is `Some` and value matches predicate.

        Examples:

        - Option is `Some` and predicate matches

        >>> Some(1).filter(lambda v: bool(v))
        Some(1)

        - Option is `Some` and predicate does not match

        >>> Some(0).filter(lambda v: bool(v))
        Nothing()

        - Predicate is ignored when option is `Nothing()`

        >>> Nothing().filter(...)
        Nothing()
        """

    @abc.abstractmethod
    def contains(self, value: T) -> bool:
        """Return true, if option is Some and container value is equal to provided value.

        Examples:

        >>> Some(1).contains(1)
        True

        >>> Some(1).contains(0)
        False

        >>> Nothing().contains(1)
        False
        """

    @abc.abstractmethod
    def zip(self, other: Option[U]) -> Option[tuple[T, U]]:
        """Zips self with another Option.

        If self is Some(s) and other is Some(o), this method returns Some((s, o)). Otherwise, Nothing is returned.

        Examples:

        >>> Some(1).zip(Some(2))
        Some((1, 2))

        >>> Some(1).zip(Nothing())
        Nothing()

        >>> Nothing().zip(Some(1))
        Nothing()
        """

    @abc.abstractmethod
    def zip_with(self, other: Option[U], func: t.Callable[[T, U], R]) -> Option[R]:
        """Zips self and another Option with function func.

        If self is Some(s) and other is Some(o), this method returns Some(func(s, o)). Otherwise, Nothing is returned.

        Examples:

        >>> Some(1).zip_with(Some(2), lambda x, y: x+y)
        Some(3)

        >>> Some(1).zip_with(Nothing(), lambda x, y: x+y)
        Nothing()

        >>> Nothing().zip_with(Some(1), lambda x, y: x+y)
        Nothing()
        """


class ResultABC(t.Generic[T, E], metaclass=abc.ABCMeta):
    __slots__ = ()

    def __init_subclass__(cls, *args: t.Any, **kwargs: t.Any):
        super().__init_subclass__(*args, **kwargs)
        abc_members = [
            member for member in dir(ResultABC) if not member.startswith("_")
        ]
        cls_members = [
            member
            for member in dir(cls)
            if not member.startswith("_")
            and not getattr(getattr(cls, member), "__isabstractmethod__", False)
        ]
        if abc_members != cls_members:
            raise TypeError(
                f"Invalid Result interface. Unknown methods: {set(cls_members).difference(abc_members) or '{}'}. Missing methods: {set(abc_members).difference(cls_members)}"
            )

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Result string representation."""

    @abc.abstractmethod
    def __eq__(self, other: t.Any) -> bool:
        """Support for `==` operator.

        Examples:

        - Two results are equal if they have the same type and hold equal values

        >>> Ok(1) == Ok(1)
        True
        >>> Err(1) == Err(1)
        True

        - A result is not equal to the value it contains

        >>> Ok(1) == 1
        False
        >>> Err(1) == 1
        False
        """

    @abc.abstractmethod
    def __ne__(self, other: t.Any) -> bool:
        """Support for `!=` operator.

        Examples:

        - Not equal always returns `True` when `other` is not a result

        >>> Ok(1) != 1
        True
        >>> Err(1) != 1
        True

        - Not equal returns `True` where `other` is not of same type

        >>> Ok(1) != Err(1)
        True
        >>> Err(1) != Ok(1)
        True

        - Not equals returns `False` where result are of same type and contain the same value

        >>> Ok(1) != Ok(1)
        False
        >>> Err(0) != Err(0)
        False
        """

    @abc.abstractmethod
    def __bool__(self) -> t.Literal[True, False]:
        """Boolean operator.

        Examples:

        - `Ok` results evaluate to `True` regardless of contained value

        >>> bool(Ok(False))
        True

        - `Err` results evaluate to `False` regardless of contained value

        >>> bool(Err(True))
        False
        """

    @abc.abstractmethod
    def __hash__(self) -> int:
        """Result hash value.

        Providing a `__hash__` method for result classes is useful
        because it allows using structures such as `set` or tools
        such as `functools.lru_cache`.

        Examples:

        - Reduce a list of 3 duplicate `Ok` results into a set of 1 `Ok` result

        >>> set([Ok(1), Ok(1), Ok(1)])
        {Ok(1)}

        - Reduce a list of 2 duplicate `Err` results into a set of 1 `Err` result

        >>> set([Err(None), Err(None), Err(None)])
        {Err(None)}
        """

    @abc.abstractmethod
    def __iter__(self: RT) -> RT:
        """Support iteration over result value.

        The returned iterator yield a single value in case of Ok
        else does not yield value.

        Examples:

        - `Ok` result behaves like an iterator yielding the contained value

        >>> list(Ok(1))
        [1]

        - `Err` result behaves like an empty iterator

        >>> list(Err(1))
        []
        """

    @abc.abstractmethod
    def __next__(self) -> T:
        """Support iteration over result value.

        Raises:
            `StopIteration`: when result is `Err` or when result is `Ok` and method is called twice.

        Examples:

        - `next` can be called once on an `Ok` result

        >>> result = Ok(1)
        >>> next(result)
        1
        >>> next(result)
        Traceback (most recent call last):
            ...
        StopIteration

        - `next` raises `StopIteration` on `Err` result

        >>> next(Err(1))
        Traceback (most recent call last):
            ...
        StopIteration
        """

    # Value property

    @property
    @abc.abstractmethod
    def value(self) -> T | E:
        """Get result value.

        Returns:
            A value of type `T` when result is `Ok[T]` or a value of type `E` when result is `Err[E]`

        Examples:

        - Access `Ok` value

        >>> Ok(1).value
        1

        - Access `Err` value

        >>> Err("BOOM").value
        'BOOM'
        """

    # Methods returning a boolean used to check type

    @abc.abstractmethod
    def is_ok(self) -> t.Literal[True, False]:
        """Check if result is `Ok`.

        Returns:
            `True` if result is `Ok` else `False`.

        Examples:

        >>> Ok().is_ok()
        True
        >>> Err().is_ok()
        False
        """

    @abc.abstractmethod
    def is_err(self) -> t.Literal[True, False]:
        """Check if result is `Err`.

        Returns:
            `True` if the result is `Err` else `False`.

        Examples:
        >>> Ok().is_err()
        False
        >>> Err().is_err()
        True
        """

    @abc.abstractmethod
    def is_ok_and(self, predicate: t.Callable[[T], bool]) -> bool:
        """Check if result is `Ok` and value inside of it matches a predicate.

        Returns:
            `True` if the result is `Ok[T]` and the value inside of it matches a predicate.

        Examples:
        >>> Ok(True).is_ok_and(lambda v: v is True)
        True
        >>> Ok(False).is_ok_and(lambda v: v is True)
        False

        - Predicate is not used and `False` is always returned when result is `Err`

        >>> Err(True).is_ok_and(...)
        False
        """

    @abc.abstractmethod
    def is_err_and(self, predicate: t.Callable[[E], bool]) -> bool:
        """Check if result is `Err` and value inside of it matches a predicate.

        Returns:
            `True` if the result is `Err[E]` and the value inside of it matches a predicate.

        Examples:

        - Result is `Err` and predicate returns `True`

        >>> Err(True).is_err_and(lambda v: v is True)
        True

        - Result is `Err` and predicate returns `False`

        >>> Err(False).is_err_and(lambda v: v is True)
        False

        - Predicate is not used and `False` is always returned when result is `Ok`

        >>> Ok(True).is_err_and(...)
        False
        """

    # Methods returning an option

    @abc.abstractmethod
    def ok(self) -> Option[T]:
        """Get an option holding `Ok` result value.

        Returns:
            `Some[T]` when result is `Ok[T]` else `Nothing()`

        Examples:

        - `Ok` results always return some value

        >>> Ok(1).ok()
        Some(1)

        - `Err` results always return `Nothing()`

        >>> Err(1).ok()
        Nothing()
        """

    @abc.abstractmethod
    def err(self) -> Option[E]:
        """Get an option holding `Err` result value.

        Returns:
            `Some[E]` when result is `Err[E]` else `Nothing()`

        Examples:

        - `Err` results always return some error value

        >>> Err(1).err()
        Some(1)

        - `Ok` results always return `Nothing()`

        >>> Ok(1).err()
        Nothing()
        """

    # Methods used to transform result value with pure function

    @abc.abstractmethod
    def map(self, func: t.Callable[[T], U]) -> Result[U, E]:
        """Maps a `Result[T, E]` to `Result[U, E]` by applying a function to `Ok` value.

        - If the result is not `Ok`, function is never called and result is forwarded.
        - If function returns a value (of type `U`),  `Ok[U]` is returned.
        - If function raises an exception, exception is raised back

        Returns:
            `Ok[U]` if result is `Ok[T]`, else forward `Err[E]`.

        Examples:

        - Chaining operations on an `Ok` result using several calls to `.map()`:

        >>> Ok(1).map(lambda x: x+1).map(lambda x: x+2)
        Ok(4)

        - Error is raised back if any mapped function raises an error:

        >>> Ok(0).map(lambda x: x/0).map(lambda x: x+2)
        Traceback (most recent call last):
            ...
        ZeroDivisionError: division by zero

        - Argument is not used when result is `Err`:

        >>> Err(1).map(lambda x: x/0)
        Err(1)
        """

    @abc.abstractmethod
    def map_err(self, func: t.Callable[[E], F]) -> Result[T, F]:
        """Maps a `Result[T, E]` to `Result[T, F]` by applying a function to `Err` value.

        - If the result is not `Err`, function is never called and result is forwarded.
        - If function returns a value (of type `F`),  `Err[F]` is returned.
        - If function raises an exception, exception is raised back

        Returns:
            `Err[F]` is `Err[T]`, else forward `Ok[T]`.

        Examples:

        - Function provided to `Result.map_err()` is only called on `Err` results

        >>> Err(1).map_err(lambda x: x+1).map(lambda x: x+2)
        Err(2)
        >>> Ok(1).map_err(lambda x: x+1).map(lambda x: x+2)
        Ok(3)

        - Error is raised back if any mapped function raises an error:

        >>> Err(0).map_err(lambda x: x/0).map(lambda x: x+2)
        Traceback (most recent call last):
            ...
        ZeroDivisionError: division by zero

        - Argument is not used when result is `Ok`:

        >>> Ok(1).map_err(...).map_err(...).map_err(...)
        Ok(1)
        """

    # Methods to swap to get back on happy path

    @abc.abstractmethod
    def swap_err(self, func: t.Callable[[E], T]) -> Ok[T]:
        """Transformes a `Err[E]` into an `Ok[T]` or forward `Ok[T]`.

        Returns
            `Ok[T]` when result is `Err[E]` or forward `Ok[T]`

        Examples:

        - Recover from error

        >>> Err(0).swap_err(lambda v: v)
        Ok(0)

        - Exceptions are raised back

        >>> Err(0).swap_err(lambda v: 1/v)
        Traceback (most recent call last):
            ...
        ZeroDivisionError: division by zero

        - Do nothing on `Ok`

        >>> Ok(0).swap_err(...)
        Ok(0)
        """

    # Methods used to execute a function without changing return value

    @abc.abstractmethod
    def inspect(self, func: t.Callable[[T], t.Any]) -> Result[T, E]:
        """Calls the provided function with the contained value only if result is `Ok` and return result.

        This function is useful when it's required to execute some callback on the ok value and there is
        no interest in the callback return value.

        Returns:
            `self`

        Examples:

        - Result is `Ok` and function is called without error

        >>> Ok(1).inspect(lambda x: None).map(lambda x: x+1)
        Ok(2)

        - Error is raised back when result is `Ok` and function raises an error

        >>> Ok(1).inspect(lambda x: x/0).map(lambda x: x+1)
        Traceback (most recent call last):
            ...
        ZeroDivisionError: division by zero

        - `Err` result is forwarded

        >>> Err(1).inspect(...)
        Err(1)
        """

    @abc.abstractmethod
    def inspect_err(self, func: t.Callable[[E], t.Any]) -> Result[T, E]:
        """Calls the provided function with the contained value only if result is `Err` and return result.

        This function is useful when it's required to execute some callback on the error value and there is
        no interest in the callback return value.

        Returns:
            `self`

        Examples:

        - Result is `Err` and function is called without error

        >>> Err(1).inspect_err(lambda x: None).map_err(lambda x: x+1)
        Err(2)

        - Exception is raised back when result is `Err` and function raises an exception

        >>> Err(1).inspect_err(lambda x: x/0).map_err(lambda x: x+1)
        Traceback (most recent call last):
            ...
        ZeroDivisionError: division by zero

        - `Ok` result is forwarded

        >>> Ok(1).inspect_err(...)
        Ok(1)
        """

    # Methods used to unwrap values

    @abc.abstractmethod
    def unwrap(self, msg: str = "") -> T:
        """Returns the contained ok value if result is `Ok` else raises an error with provided message.

        Examples:

        - Expecting on a `Ok` value returns the contained value

        >>> Ok(1).unwrap("expected ok result")
        1

        - Expecting on `Err` result raises a `ErrResultError`

        >>> Err(1).unwrap("expected ok result")
        Traceback (most recent call last):
            ...
        results.safe.ErrResultError: expected ok result
        """

    @abc.abstractmethod
    def unwrap_err(self, msg: str = "") -> E:
        """Returns the contained error value if result is `Ok` else raises an error with provided message.

        Examples:

        - Expecting error on a `Err` value returns the contained value

        >>> Err(1).unwrap_err("expected err result")
        1

        - Expecting error on `Ok` result raises a `OkResultError`

        >>> Ok(1).unwrap_err("expected err result")
        Traceback (most recent call last):
            ...
        results.safe.OkResultError: expected err result
        """

    # Methods used to unwrap values with default

    @abc.abstractmethod
    def unwrap_or(self, default: T) -> T:
        """Get `Ok` result value or default value.

        Returns:
            the contained Ok value if result is `Ok` or a provided default.

        Examples:

        - Get value from `Ok` result

        >>> Ok(1).unwrap_or(2)
        1

        - Get default value from `Err` result

        >>> Err(1).unwrap_or(2)
        2
        """

    @abc.abstractmethod
    def unwrap_or_else(self, default: t.Callable[[E], T]) -> T:
        """Get `Ok` result value or default value from gien callable

        Returns:
            the result value if result is `Ok` or a provided default computed using given callable if result is `Err`

        Examples:

        - Get value from `Ok` result

        >>> Ok(1).unwrap_or_else(...)
        1

        - Get default value from `Err` result

        >>> Err(1).unwrap_or_else(lambda err: err + 1)
        2
        """

    # Methods used to acess other result conditionally

    @abc.abstractmethod
    def and_result(self, other: Result[U, E]) -> Result[U, E]:
        """Returns other only if result is `Ok`, else forward `Err[E]`.

        Returns:
            `Result[U, E]` when result is `Ok`, else forward `Err[E]`

        Examples:

        - Provided argument is always returned when result is `Ok`

        >>> Ok(1).and_result(Ok(2))
        Ok(2)
        >>> Ok(1).and_result(Err(1))
        Err(1)

        - Result is always forward when `Err`

        >>> Err(1).and_result(Ok(1))
        Err(1)
        """

    @abc.abstractmethod
    def or_result(self, other: Result[T, F]) -> Result[T, F]:
        """Returns other only if result is `Err`, else forward `Ok[T]`.

        Returns:
            `Result[T, F]` when result is `Err`, else forward `Ok[T]`

        Examples:

        - Provided argument is always returned when result is `Err`

        >>> Err(1).or_result(Ok(2))
        Ok(2)
        >>> Err(1).or_result(Err(2))
        Err(2)

        - Result is always forward when `Ok`

        >>> Ok(1).or_result(Err(1))
        Ok(1)
        """

    # Methods used to transform result

    @abc.abstractmethod
    def bind(self, func: t.Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Transform an `Ok[T]` result by applying given function.

        The provided function must accept a value and returns a result.

        If the function returns a value (and not a result), use `Result.map()` method instead.

        Returns:
            `Result[U, E]` if result is `Ok` else forward `Err`.

        Examples:

        - Transform an `Ok` result into an `Ok` result

        >>> Ok(1).bind(lambda x: Ok(x+1)).map(lambda x: x*2)
        Ok(4)

        - Transform an `Ok` result into an `Err` result

        >>> Ok(1).bind(lambda x: Err(x+1)).map(lambda x: x*2)
        Err(2)

        - Error is raised back

        >>> Ok(1).bind(lambda x: x/0)
        Traceback (most recent call last):
            ...
        ZeroDivisionError: division by zero

        - `Err` results are left untouched

        >>> Err(1).bind(lambda x: Ok(x+1)).map(lambda x: x*2)
        Err(1)
        """

    @abc.abstractmethod
    def bind_err(self, func: t.Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Transform an `Err[E]` result by applying given function.

        The provided function must accept an error value and returns a result.

        If the function returns a value (and not a result), use `Result.map_err()` method instead.

        Returns:
            `Result[T, F]` if result is `Err` else forward `Ok[T]`.

        Examples:

        - Transform an `Err` result into an `Err` result

        >>> Err(1).bind_err(lambda x: Err(x+1)).map_err(lambda x: x*2)
        Err(4)

        - Transform an `Err` result into an `Ok` result

        >>> Err(1).bind_err(lambda x: Ok(x+1)).map_err(lambda x: x*2)
        Ok(2)

        - `Ok` results are left untouched

        >>> Ok(1).bind_err(lambda x: Ok(x+1)).map_err(lambda x: x*2)
        Ok(1)
        """

    @abc.abstractmethod
    def bind_result(
        self, func: t.Callable[[Result[T, E]], Result[U, F]]
    ) -> Result[U, F]:
        """Transform a `Result[T, E]` result by applying given function.

        The provided function must accept a result and return a result.

        Examples:

        - Function must accept a result and return a result

        >>> Ok(1).bind_result(lambda r: Err(r.unwrap()))
        Err(1)

        - Function is ignored when result result

        >>> Err(1).bind_result(lambda r: Ok(r.unwrap_err()))
        Ok(1)
        """

    # Methods used to check equality with result value

    @abc.abstractmethod
    def contains(self, value: t.Any) -> bool:
        """Check if result is `Ok` and contains given value.

        Returns
            `True` if the result is an `Ok` result containing the given value, else `False`.

        Examples:

        - `Ok` result may contain value

        >>> Ok(1).contains(1)
        True
        >>> Ok(1).contains(0)
        False

        - `Err` result never contains value

        >>> Err(1).contains(1)
        False
        """

    @abc.abstractmethod
    def contains_err(self, value: t.Any) -> bool:
        """Check if result is `Err` and contains given value.

        Returns
            `True` if the result is an `Err` result containing the given error value, else `False`.

        Examples:

        - `Err` result may contain error value

        >>> Err(1).contains_err(1)
        True
        >>> Err(1).contains_err(0)
        False

        - `Ok` result never contains error value

        >>> Ok(1).contains_err(1)
        False
        """


class Some(OptionABC[T]):
    """
    A value that indicates presence of data and which stores arbitrary data for the return value.
    """

    _value: T
    __slots__ = ("_value", "_visited")
    __match_args__ = ("value",)

    @t.overload
    def __init__(self: Some[bool]) -> None:
        ...  # pragma: no cover

    @t.overload
    def __init__(self, value: T) -> None:
        ...  # pragma: no cover

    def __init__(self, value: t.Any = True) -> None:
        self._value = value
        self._visited = False

    @property
    def value(self) -> T:
        return self._value

    def __repr__(self) -> str:
        return "Some({})".format(repr(self._value))

    def __eq__(self, other: t.Any) -> bool:
        return isinstance(other, Some) and self.value == other.value

    def __ne__(self, other: t.Any) -> bool:
        return not (self == other)  # NOSONAR

    def __bool__(self) -> t.Literal[True]:
        return True

    def __hash__(self) -> int:
        return hash((True, self._value))

    def __iter__(self) -> t.Iterator[T]:
        self._visited = False
        return self

    def __next__(self) -> T:
        if not self._visited:
            self._visited = True
            return self._value
        raise StopIteration

    def is_some(self) -> t.Literal[True]:
        return True

    def is_some_and(self, predicate: t.Callable[[T], bool]) -> bool:
        return predicate(self._value)

    def is_nothing(self) -> t.Literal[False]:
        return False

    def unwrap(self, msg: str = "") -> T:
        return self._value

    def unwrap_nothing(self, msg: str = "") -> t.NoReturn:
        raise SomeOptionError(self, msg)

    def unwrap_or(self, default: object) -> T:
        return self._value

    def unwrap_or_else(self, func: object) -> T:
        return self._value

    def map(self, func: t.Callable[[T], U]) -> Some[U]:
        return Some(func(self._value))

    def inspect(self, func: t.Callable[[T], t.Any]) -> Some[T]:
        func(self._value)
        return self

    def ok_or(self, err: object) -> Ok[T]:
        return Ok(self._value)

    def ok_or_else(self, err: object) -> Ok[T]:
        return Ok(self._value)

    def and_option(self, other: Option[U]) -> Option[U]:
        if other:
            return other
        return NOTHING

    def bind(self, func: t.Callable[[T], OT]) -> OT:
        return func(self._value)

    def filter(self, predicate: t.Callable[[T], bool]) -> Option[T]:
        return self if predicate(self._value) else NOTHING

    def or_option(self, other: object) -> Some[T]:
        return self

    def or_else(self, func: object) -> Some[T]:
        return self

    def xor_option(self, other: Option[T]) -> Option[T]:
        return NOTHING if other else self

    def contains(self, value: T) -> bool:
        return self._value == value

    def zip(self, other: Option[U]) -> Option[tuple[T, U]]:
        if other:
            return Some((self._value, other._value))
        return NOTHING

    def zip_with(self, other: Option[U], func: t.Callable[[T, U], R]) -> Option[R]:
        if other:
            return Some(func(self._value, other._value))
        return NOTHING


class Nothing(OptionABC[t.NoReturn], metaclass=SingletonABC):
    """
    A value that signifies failure and which stores arbitrary data for the error.
    """

    @property
    def value(self) -> t.NoReturn:
        raise NothingOptionError("Nothing objects do not have value")

    def __repr__(self) -> str:
        return "Nothing()"

    def __eq__(self, other: t.Any) -> bool:
        return self is other

    def __ne__(self, other: t.Any) -> bool:
        return self is not other

    def __hash__(self) -> int:
        return hash((False, "Nothing"))

    def __bool__(self) -> t.Literal[False]:
        return False

    def __iter__(self) -> t.Iterator[t.NoReturn]:
        return self

    def __next__(self) -> t.NoReturn:
        raise StopIteration

    def is_some(self) -> t.Literal[False]:
        return False

    def is_some_and(self, predicate: t.Callable[[t.Any], bool]) -> t.Literal[False]:
        return False

    def is_nothing(self) -> t.Literal[True]:
        return True

    def unwrap(self, msg: str = "") -> t.NoReturn:
        raise NothingOptionError(msg)

    def unwrap_nothing(self, msg: str = "") -> None:
        return None

    def unwrap_or(self, default: U) -> U:
        return default

    def unwrap_or_else(self, func: t.Callable[[], U]) -> U:
        return func()

    def map(self, func: object) -> Nothing:
        return self

    def inspect(self, func: object) -> Nothing:
        return self

    def ok_or(self, err: F) -> Err[F]:
        return Err(err)

    def ok_or_else(self, err: t.Callable[[], E]) -> Err[E]:
        return Err(err())

    def and_option(self, other: object) -> Nothing:
        return self

    def bind(self, func: object) -> Nothing:
        return self

    def filter(self, predicate: object) -> Nothing:
        return self

    def or_option(self, other: OT) -> OT:
        return other

    def or_else(self, func: t.Callable[[], OT]) -> OT:
        return func()

    def xor_option(self, other: OT) -> OT:
        return other

    def contains(self, other: object) -> t.Literal[False]:
        return False

    def zip(self, other: object) -> Nothing:
        return self

    def zip_with(self, other: object, func: object) -> Nothing:
        return self


class Ok(ResultABC[T, t.NoReturn]):
    """
    A value that indicates success and which stores arbitrary data for the return value.
    """

    _value: T
    __slots__ = ("_value", "_visited")
    __match_args__ = ("value",)

    @t.overload
    def __init__(self, value: T) -> None:
        ...  # pragma: no cover

    @t.overload
    def __init__(self: Ok[bool]) -> None:
        ...  # pragma: no cover

    def __init__(self, value: t.Any = True) -> None:
        self._value = value
        self._visited = False

    @property
    def value(self) -> T:
        return self._value

    def __repr__(self) -> str:
        return "Ok({})".format(repr(self._value))

    def __eq__(self, other: t.Any) -> bool:
        return isinstance(other, Ok) and self.value == other.value

    def __ne__(self, other: t.Any) -> bool:
        return not (self == other)  # NOSONAR

    def __bool__(self) -> t.Literal[True]:
        return True

    def __hash__(self) -> int:
        return hash((True, self._value))

    def __iter__(self) -> Ok[T]:
        self._visited = False
        return self

    def __next__(self) -> T:
        if self._visited:
            raise StopIteration
        self._visited = True
        return self._value

    def is_ok(self) -> t.Literal[True]:
        return True

    def is_ok_and(self, predicate: t.Callable[[T], bool]) -> bool:
        return predicate(self._value)

    def is_err(self) -> t.Literal[False]:
        return False

    def is_err_and(self, predicate: object) -> t.Literal[False]:
        return False

    def ok(self) -> Some[T]:
        return Some(self._value)

    def err(self) -> Nothing:
        return NOTHING

    def map(self, func: t.Callable[[T], U]) -> Ok[U]:
        return Ok(func(self._value))

    def map_err(self, func: object) -> Ok[T]:
        return self

    def swap_err(self, value: object) -> Ok[T]:
        return self

    def inspect(self, func: t.Callable[[T], t.Any]) -> Ok[T]:
        func(self._value)
        return self

    def inspect_err(self, func: object) -> Ok[T]:
        return self

    def unwrap(self, msg: str = "") -> T:
        """
        Return the value.
        """
        return self._value

    def unwrap_err(self, msg: str = "") -> t.NoReturn:
        raise OkResultError(self, msg)

    def and_result(self, other: Result[U, E]) -> Result[U, E]:
        return other

    def or_result(self, other: object) -> Ok[T]:
        return self

    def bind(self, func: t.Callable[[T], Result[U, E]]) -> Result[U, E]:
        return func(self._value)

    def bind_err(self, other: object) -> Ok[T]:
        return self

    def bind_result(self, func: t.Callable[[Ok[T]], Result[U, F]]) -> Result[U, F]:
        return func(self)

    def unwrap_or(self, default: object) -> T:
        return self._value

    def unwrap_or_else(self, default: object) -> T:
        return self._value

    def contains(self, value: T) -> bool:
        return self._value == value

    def contains_err(self, value: object) -> t.Literal[False]:
        return False


ErrT = t.TypeVar("ErrT", bound="Err[t.Any]")


class Err(ResultABC[t.NoReturn, E]):
    """
    A value that signifies failure and which stores arbitrary data for the error.
    """

    _value: E
    __slots__ = "_value"
    __match_args__ = ("value",)

    @t.overload
    def __init__(self, value: E) -> None:
        ...  # pragma: no cover

    @t.overload
    def __init__(self: Err[bool]) -> None:
        ...  # pragma: no cover

    def __init__(self, value: t.Any = False) -> None:
        self._value = value

    @property
    def value(self) -> E:
        return self._value

    def __repr__(self) -> str:
        return "Err({})".format(repr(self._value))

    def __eq__(self, other: t.Any) -> bool:
        return isinstance(other, Err) and self._value == other._value

    def __ne__(self, other: t.Any) -> bool:
        return not (self == other)  # NOSONAR

    def __hash__(self) -> int:
        return hash((False, self._value))

    def __bool__(self) -> t.Literal[False]:
        return False

    def __iter__(self: ErrT) -> ErrT:
        return self

    def __next__(self) -> t.NoReturn:
        raise StopIteration

    def is_ok(self) -> t.Literal[False]:
        return False

    def is_ok_and(self, predicate: object) -> t.Literal[False]:
        return False

    def is_err(self) -> t.Literal[True]:
        return True

    def is_err_and(self, predicate: t.Callable[[E], bool]) -> bool:
        return predicate(self._value)

    def ok(self) -> Nothing:
        return NOTHING

    def err(self) -> Some[E]:
        return Some(self._value)

    def map(self, func: object) -> Err[E]:
        return self

    def map_err(self, func: t.Callable[[E], F]) -> Err[F]:
        return Err(func(self._value))

    def inspect(self, func: object) -> Err[E]:
        return self

    def inspect_err(self, func: t.Callable[[E], t.Any]) -> Err[E]:
        func(self._value)
        return self

    def unwrap(self, msg: str = "") -> t.NoReturn:
        raise ErrResultError(self, msg)

    def unwrap_err(self, msg: str = "") -> E:
        return self._value

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, default: t.Callable[[E], T]) -> T:
        return default(self._value)

    def and_result(self, other: object) -> Err[E]:
        return self

    def or_result(self, other: Result[T, F]) -> Result[T, F]:
        return other

    def bind(self, other: object) -> Err[E]:
        return self

    def bind_err(self, func: t.Callable[[E], Result[T, F]]) -> Result[T, F]:
        return func(self._value)

    def bind_result(self, func: t.Callable[[Err[E]], Result[U, F]]) -> Result[U, F]:
        return func(self)

    def contains(self, value: object) -> t.Literal[False]:
        return False

    def contains_err(self, value: object) -> bool:
        return self._value == value

    def swap_err(self, func: t.Callable[[E], T]) -> Ok[T]:
        return Ok(func(self._value))


# define Result as a generic type alias for use
# in type annotations
Result: TypeAlias = t.Union[Ok[T], Err[E]]
"""
A simple `Result` type inspired by Rust (https://doc.rust-lang.org/std/result/enum.Result.html)
"""

ResultType: Final = (Ok, Err)
"""
A type to use in `isinstance` checks.
This is purely for convenience sake, as you could also just write `isinstance(res, (Ok, Err))
"""


# define Option as a generic type alias for use
# in type annotations
Option: TypeAlias = t.Union[Some[T], Nothing]
"""
A simple `Option` type inspired by Rust (https://doc.rust-lang.org/std/option/enum.Option.html).
"""

OptionType: Final = (Some, Nothing)
"""
A type to use in `isinstance` checks.
This is purely for convenience sake, as you could also just write `isinstance(res, (Some, Nothing))
"""

# create Nothing() singleton
NOTHING = Nothing()


# Define errors
class UnwrapError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class OptionError(UnwrapError):
    _option: Option[t.Any]

    def __init__(self, option: Option[t.Any], message: str) -> None:
        self._option = option
        super().__init__(message)

    @property
    def option(self) -> Option[t.Any]:
        """Returns the original option"""
        return self._option


class ResultError(UnwrapError):
    _result: Result[t.Any, t.Any]

    def __init__(self, result: Result[t.Any, t.Any], message: str) -> None:
        self._result = result
        super().__init__(message)

    @property
    def result(self) -> Result[t.Any, t.Any]:
        """Returns the original result."""
        return self._result


class SomeOptionError(OptionError):
    _option: Some[t.Any]

    def __init__(self, option: Some[t.Any], message: str) -> None:
        super().__init__(option, message)


class NothingOptionError(OptionError):
    _option: Nothing

    def __init__(self, message: str) -> None:
        super().__init__(NOTHING, message)


class OkResultError(ResultError):
    _result: Ok[t.Any]

    def __init__(self, result: Ok[t.Any], message: str) -> None:
        super().__init__(result, message)


class ErrResultError(ResultError):
    _result: Err[t.Any]

    def __init__(self, result: Err[t.Any], message: str) -> None:
        super().__init__(result, message)
