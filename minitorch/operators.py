"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -1.0 * x


def lt(x: float, y: float) -> float:
    return float(x < y)


def eq(x: float, y: float) -> float:
    return float(x == y)


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def relu(x: float) -> float:
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    return math.log(x + EPS)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d / (x + EPS)


def inv(x: float) -> float:
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], list[float]]:
    def apply(ls: Iterable[float]) -> list[float]:
        return [fn(x) for x in ls]

    return apply


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], list[float]]:
    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> list[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return apply


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    def apply(ls: Iterable[float]) -> float:
        value = start
        for x in ls:
            value = fn(value, x)
        return value

    return apply


negList = map(neg)
addLists = zipWith(add)
sum = reduce(add, 0.0)
prod = reduce(mul, 1.0)
