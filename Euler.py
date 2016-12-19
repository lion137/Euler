from math import sqrt, ceil
import random
import itertools
import operator as op
from functools import reduce


# --- fast expotentation-------------------------------------------------------------------------
def exp(x, n):
    if n == 0:
        return 1
    elif n == 1:
        return x
    elif even(n):
        a = exp(x * x, n // 2)
        return a
    elif odd(n):
        b = x * exp(x * x, (n - 1) // 2)
        return b


fact = (1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880)


# def factorial(n): return reduce(lambda x,y:x*y,range(1,n+1),1)
def factorial(n): return reduce(op.mul, range(1, n + 1))


def is_perm(a, b): return sorted(str(a)) == sorted(str(b))


def is_palindromic(n): n = str(n); return n == n[::-1]


def is_pandigital(n, s=9): n = str(n); return len(n) == s and not '1234567890'[:s].strip(n)



# --- Calculate the sum of proper divisors for n--------------------------------------------------
def d(n):
    s = 1
    t = sqrt(n)
    for i in range(2, int(t) + 1):
        if n % i == 0: s += i + n / i
    if t == int(t): s -= t  # correct s if t is a perfect square
    return s


# --- Create a list of all palindromic numbers with k digits--------------------------------------
def pal_list(k):
    if k == 1:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9]
    return [sum([n * (10 ** i) for i, n in enumerate(([x] + list(ys) + [z] + list(ys)[::-1] + [x]) if k % 2
                                                     else ([x] + list(ys) + list(ys)[::-1] + [x]))])
            for x in range(1, 10)
            for ys in itertools.product(range(10), repeat=k / 2 - 1)
            for z in (range(10) if k % 2 else (None,))]


# --- sum of factorial's digits-------------------------------------------------------------------
def sof_digits(n):
    s = 0
    while n > 0:
        s, n = s + fact[n % 10], n // 10
    return s


# --- find the nth Fibonacci number---------------------------------------------------------------
def fibonacci(n):
    """
    Find the nth number in the Fibonacci series.  Example:
    
    >>>fibonacci(100)
    354224848179261915075

    Algorithm & Python source: Copyright (c) 2013 Nayuki Minase
    Fast doubling Fibonacci algorithm

http://nayuki.eigenstate.org/page/fast-fibonacci-algorithms

    """
    if n < 0:
        raise ValueError("Negative arguments not implemented")
    return _fib(n)[0]


# Returns a tuple (F(n), F(n+1))
def _fib(n):
    if n == 0:
        return (0, 1)
    else:
        a, b = _fib(n // 2)
        c = a * (2 * b - a)
        d = b * b + a * a
        if n % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)


# --- sum of digits -----------------------------------------------------------------------------

def sum_digits(n):
    s = 0
    while n > 0:
        s, n = s + (n % 10), n // 10
    return s


# --- sum of squares of digits-------------------------------------------------------------------
def sos_digits(n):
    s = 0
    while n > 0:
        s, n = s + (n % 10) ** 2, n // 10
    return s


# --- sum of the digits to a power e-------------------------------------------------------------
def pow_digits(n, e):
    s = 0
    while n > 0:
        s, n = s + (n % 10) ** e, n // 10
    return s


# --- check n for prime--------------------------------------------------------------------------
def is_prime(n):
    n = int(n)
    if n == 2 or n == 3: return True
    if n < 2 or n % 2 == 0: return False
    if n < 9: return True
    if n % 3 == 0: return False
    r = int(sqrt(n))
    f = 5
    while f <= r:
        if n % f == 0: return False
        if n % (f + 2) == 0: return False
        f += 6
    return True


# --- Miller-Rabin primality test----------------------------------------------------------------
def miller_rabin(n):
    """
    Check n for primalty:  Example:

    >miller_rabin(162259276829213363391578010288127)    #Mersenne prime #11
    True

    Algorithm & Python source:

http://en.literateprograms.org/Miller-Rabin_primality_test_(Python)

    """
    d = n - 1
    s = 0
    while d % 2 == 0:
        d >>= 1
        s += 1
    for repeat in range(20):
        a = 0
        while a == 0:
            a = random.randrange(n)
        if not miller_rabin_pass(a, s, d, n):
            return False
    return True


def miller_rabin_pass(a, s, d, n):
    a_to_power = pow(a, d, n)
    if a_to_power == 1:
        return True
    for i in range(s - 1):
        if a_to_power == n - 1:
            return True
        a_to_power = (a_to_power * a_to_power) % n
    return a_to_power == n - 1


# --- factor a number into primes and frequency----------------------------------------------------
def factor(n):
    """
    find the prime factors of n along with their frequencies. Example:

    >>> factor(786456)
    [(2,3), (3,3), (11,1), (331,1)]
    """
    if n in [-1, 0, 1]: return []
    if n < 0: n = -n
    F = []
    while n != 1:
        p = trial_division(n)
        e = 1
        n /= p
        while n % p == 0:
            e += 1;
            n /= p
        F.append((p, e))
    F.sort()
    return F


def trial_division(n, bound=None):
    if n == 1: return 1
    for p in [2, 3, 5]:
        if n % p == 0: return p
    if bound == None: bound = n
    dif = [6, 4, 2, 4, 2, 4, 6, 2]
    m = 7;
    i = 1
    while m <= bound and m * m <= n:
        if n % m == 0:
            return m
        m += dif[i % 8]
        i += 1
    return n


# --- greatest common divisor----------------------------------------------------------------------
def gcd(a, b):
    """
    Compute the greatest common divisor of a and b. Examples:
    
    >>> gcd(14, 15)    #co-prime
    1
    >>> gcd(5*5, 3*5)
    5
    """
    if a < 0:  a = -a
    if b < 0:  b = -b
    if a == 0: return b
    while b != 0:
        a, b = b, a % b
    return a


# --- binomial coefficients-----------------------------------------------------------------------
def binomial(n, k):
    """
    Calculate C(n,k), the number of ways can k be chosen from n. Example:
    
    >>>binomial(30,12)
    86493225
    """
    if k == 0 or n == k: return 1
    nt = 1
    for i in range(1, k + 1):
        nt = nt * (n - i + 1) // i
    return nt


# --- catalan number------------------------------------------------------------------------------
def catalan_number(n):
    """
    Calculate the nth Catalan number. Example:
    
    >>>catalan_number(10)
    16796
    """
    nm = dm = 1
    for k in range(2, n + 1):
        nm, dm = (nm * (n + k), dm * k)
    return nm / dm


# --- generate prime numbers----------------------------------------------------------------------
def prime_sieve(n):
    """
    Return a list of prime numbers from 2 to a prime < n. Very fast (n<10,000,000) in 0.4 sec.
    
    Example:
    >>>prime_sieve(25)
    [2, 3, 5, 7, 11, 13, 17, 19, 23]

    Algorithm & Python source: Robert William Hanks

http://stackoverflow.com/questions/17773352/python-sieve-prime-numbers

    """
    sieve = [True] * (n // 2)
    for i in range(3, int(n ** 0.5) + 1, 2):
        if sieve[i // 2]:
            sieve[i * i // 2::i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
    return [2] + [2 * i + 1 for i in range(1, n // 2) if sieve[i]]


# --- bezout coefficients--------------------------------------------------------------------------
def bezout(a, b):
    """
    Bezout coefficients (u,v) of (a,b) as:

        a*u + b*v = gcd(a,b)

    Result is the tuple: (u, v, gcd(a,b)). Examples:

    >>> bezout(7*3, 15*3)
    (-2, 1, 3)
    >>> bezout(24157817, 39088169)    #sequential Fibonacci numbers
    (-14930352, 9227465, 1)

    Algorithm source: Pierre L. Douillet

http://www.douillet.info/~douillet/working_papers/bezout/node2.html

    """
    u, v, s, t = 1, 0, 0, 1
    while b != 0:
        q, r = divmod(a, b)
        a, b = b, r
        u, s = s, u - q * s
        v, t = t, v - q * t

    return (u, v, a)


# returns list of digits of x:
def digits(x):
    if x == 0:
        return 0
    T = []
    for n in range(1000):
        if x > 0:
            T.append(x % 10)
            x = x // 10
        else:
            break
    return T


# Rotate once right a numberL

def rot_right(x):
    l = number_of_digits(x)
    tmp1 = eu.exp(10, l - 1)
    b = x % 10
    a = x // 10
    return a + b * tmp1


# Rotate a list arg: list and numbers of rotation - up to len -1
def rotate(xs, n):
    return xs[-n:] + xs[:-n]


# Create a list of 0 - 9 pandigotals

def createPandigitals():  # cretaes 0-9 pandigitals
    Pan = []
    Num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for x in Num[1:]:
        Num1 = Num[:Num.index(x)] + Num[Num.index(x) + 1:]
        for x1 in Num1:
            Num2 = Num1[:Num1.index(x1)] + Num1[Num1.index(x1) + 1:]
            for x2 in Num2:
                Num3 = Num2[:Num2.index(x2)] + Num2[Num2.index(x2) + 1:]
                for x3 in Num3:
                    Num4 = Num3[:Num3.index(x3)] + Num3[Num3.index(x3) + 1:]
                    for x4 in Num4:
                        Num5 = Num4[:Num4.index(x4)] + Num4[Num4.index(x4) + 1:]
                        for x5 in Num5:
                            Num6 = Num5[:Num5.index(x5)] + Num5[Num5.index(x5) + 1:]
                            for x6 in Num6:
                                Num7 = Num6[:Num6.index(x6)] + Num6[Num6.index(x6) + 1:]
                                for x7 in Num7:
                                    Num8 = Num7[:Num7.index(x7)] + Num7[Num7.index(x7) + 1:]
                                    for x8 in Num8:
                                        Num9 = Num8[:Num8.index(x8)] + Num8[Num8.index(x8) + 1:]
                                        for x9 in Num9:
                                            T = [x, x1, x2, x3, x4, x5, x6, x7, x8, x9]
                                            Pan.append(T)
    return Pan


# --- filter list return primes---------------------------------------
# --- primes_filter(xs)-----------------------------------------------
def primes_filter(xs):
    T = []
    for x in xs:
        if miller_rabin(x):
            T.append(x)
    return T


def extendedeuclicid(a, b):
    """ Extended Euclicid Algorithm, returns gcd, and x, y in d = gcd(a, b) = ax + by
        complexity: O(lg(b)), and O(beta^3) bit operations (call it on 2 beta bits numbers) assuming that

        mult and division is O(beta^2)"""
    if b == 0:
        return [a, 1, 0]
    else:
        [dn, xn, yn] = extendedeuclicid(b, a % b)
        [d, x, y] = [dn, yn, xn - (a // b) * yn]
        return [d, x, y]

def modular_linear_eq_solver(a, b, n):
    """solved two modular linear equations"""
    result = []
    [d, xp, yp] = extendedeuclicid(a, n)
    if b % d == 0:
        x0 = (xp * (b // d)) % n
        for i in range(0, d):
            result.append((x0 + i*(n // d)) % n)
    else:
        result = ["no solutions"]
    return result

def modularexponenation(base, exponent, modulus):
    """modular exponentiation"""
    if modulus == 1:
        return 0
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent >>= 1
        base = (base * base) % modulus
    return result

def toStr(n, base):
    """convert n tos str in base base"""
    convertString = "0123456789ABCDEF"
    if n < base:
        return convertString[n]
    else:
        return toStr(n // base, base) + convertString[n % base]
