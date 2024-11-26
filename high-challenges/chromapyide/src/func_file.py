# Defining the entire list of function bodies without printing to screen.

function_bodies = [
    """
def add(a, b):
    return a + b
""",
    """
def subtract(a, b):
    return a - b
""",
    """
def multiply(a, b):
    return a * b
""",
    """
def divide(a, b):
    return a / b if b != 0 else 'Division by zero'
""",
    """
def modulus(a, b):
    return a % b
""",
    """
def power(a, b):
    return a ** b
""",
    """
def floor_divide(a, b):
    return a // b
""",
    """
def negate(a):
    return -a
""",
    """
def is_even(a):
    return a % 2 == 0
""",
    """
def is_odd(a):
    return a % 2 != 0
""",
    """
def absolute_value(a):
    return abs(a)
""",
    """
def max_of_two(a, b):
    return max(a, b)
""",
    """
def min_of_two(a, b):
    return min(a, b)
""",
    """
def swap(a, b):
    return b, a
""",
    """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
""",
    """
def is_positive(a):
    return a > 0
""",
    """
def is_negative(a):
    return a < 0
""",
    """
def square(a):
    return a ** 2
""",
    """
def cube(a):
    return a ** 3
""",
    """
def square_root(a):
    return a ** 0.5
""",
    """
def cube_root(a):
    return a ** (1/3)
""",
    """
def is_divisible(a, b):
    return a % b == 0
""",
    """
import math
def gcd(a, b):
    return math.gcd(a, b)
""",
    """
def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)
""",
    """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
""",
    """
def sum_list(lst):
    return sum(lst)
""",
    """
def product_list(lst):
    result = 1
    for num in lst:
        result *= num
    return result
""",
    """
def average(lst):
    return sum(lst) / len(lst)
""",
    """
def max_in_list(lst):
    return max(lst)
""",
    """
def min_in_list(lst):
    return min(lst)
""",
    """
def reverse_string(s):
    return s[::-1]
""",
    """
def is_palindrome(s):
    return s == s[::-1]
""",
    """
def string_length(s):
    return len(s)
""",
    """
def to_uppercase(s):
    return s.upper()
""",
    """
def to_lowercase(s):
    return s.lower()
""",
    """
def capitalize_words(s):
    return s.title()
""",
    """
def find_in_list(lst, value):
    return lst.index(value) if value in lst else -1
""",
    """
def remove_duplicates(lst):
    return list(set(lst))
""",
    """
def union_sets(set1, set2):
    return set1.union(set2)
""",
    """
def intersect_sets(set1, set2):
    return set1.intersection(set2)
""",
    """
def difference_sets(set1, set2):
    return set1.difference(set2)
""",
    """
def sort_list(lst):
    return sorted(lst)
""",
    """
import random
def shuffle_list(lst):
    random.shuffle(lst)
    return lst
""",
    """
def even_numbers(n):
    return [i for i in range(n+1) if i % 2 == 0]
""",
    """
def odd_numbers(n):
    return [i for i in range(n+1) if i % 2 != 0]
""",
    """
def list_length(lst):
    return len(lst)
""",
    """
def in_range(n, start, end):
    return start <= n <= end
""",
    """
def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib
""",
    """
def sum_of_digits(n):
    return sum([int(digit) for digit in str(n)])
""",
    """
def binary_to_decimal(b):
    return int(b, 2)
""",
    """
def decimal_to_binary(n):
    return bin(n)[2:]
""",
    """
def celsius_to_fahrenheit(c):
    return (c * 9/5) + 32
""",
    """
def fahrenheit_to_celsius(f):
    return (f - 32) * 5/9
""",
    """
def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
""",
    """
def count_vowels(s):
    return sum(1 for char in s.lower() if char in 'aeiou')
""",
    """
def count_consonants(s):
    return sum(1 for char in s.lower() if char.isalpha() and char not in 'aeiou')
""",
    """
def remove_spaces(s):
    return s.replace(" ", "")
""",
    """
def is_anagram(s1, s2):
    return sorted(s1) == sorted(s2)
""",
    """
def factorial_iterative(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result
""",
    """
def is_perfect_square(n):
    return int(n ** 0.5) ** 2 == n
""",
    """
def is_armstrong(n):
    return sum(int(digit) ** len(str(n)) for digit in str(n)) == n
""",
    """
def reverse_list(lst):
    return lst[::-1]
""",
    """
def sum_of_squares(n):
    return sum(i**2 for i in range(1, n+1))
""",
    """
def sum_of_cubes(n):
    return sum(i**3 for i in range(1, n+1))
""",
    """
def unique_elements(lst):
    return list(set(lst))
""",
    """
def second_largest(lst):
    unique_lst = list(set(lst))
    unique_lst.sort()
    return unique_lst[-2] if len(unique_lst) >= 2 else None
""",
    """
def merge_lists(lst1, lst2):
    return lst1 + lst2
""",
    """
def common_elements(lst1, lst2):
    return list(set(lst1) & set(lst2))
""",
    """
def count_occurrences(lst, element):
    return lst.count(element)
""",
    """
def is_subset(lst1, lst2):
    return set(lst1).issubset(set(lst2))
""",
    """
def first_n_primes(n):
    primes = []
    i = 2
    while len(primes) < n:
        if all(i % p != 0 for p in primes):
            primes.append(i)
        i += 1
    return primes
""",
    """
def count_words(s):
    return len(s.split())
""",
    """
def count_sentences(s):
    return len(s.split('.'))
""",
    """
def longest_word(s):
    words = s.split()
    return max(words, key=len)
""",
    """
def reverse_words(s):
    return ' '.join(reversed(s.split()))
""",
    """
def sort_words(s):
    return ' '.join(sorted(s.split()))
""",
    """
def sum_matrix(matrix):
    return sum(sum(row) for row in matrix)
""",
    """
def transpose_matrix(matrix):
    return [list(row) for row in zip(*matrix)]
""",
    """
def matrix_multiply(matrix1, matrix2):
    result = [[sum(a * b for a, b in zip(row, col)) for col in zip(*matrix2)] for row in matrix1]
    return result
""",
    """
def mean_of_matrix(matrix):
    return sum_matrix(matrix) / (len(matrix) * len(matrix[0]))
""",
    """
def flatten_matrix(matrix):
    return [element for row in matrix for element in row]
""",
    """
def convert_to_int_list(lst):
    return list(map(int, lst))
""",
    """
def convert_to_float_list(lst):
    return list(map(float, lst))
""",
    """
def unique_words(s):
    return list(set(s.split()))
""",
    """
def extract_digits(s):
    return [int(char) for char in s if char.isdigit()]
""",
    """
def convert_list_to_string(lst):
    return ''.join(lst)
""",
    """
import random
def generate_random_number(start, end):
    return random.randint(start, end)
""",
    """
import random
def generate_random_float(start, end):
    return random.uniform(start, end)
""",
    """
def convert_to_roman(num):
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syb = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ''
    for i in range(len(val)):
        count = num // val[i]
        roman_num += syb[i] * count
        num -= val[i] * count
    return roman_num
"""
]

# Implement the functions described in functions_explanations_part10
function_bodies_10 = [
"""
def add_to_string(value, string):
    if 0 <= value <= 255:
        return string + chr(value)
    else:
        raise ValueError("Value must be in the range 0 to 255")
""",
"""
def concat_strings(global_array,str1, str2):
    global_array.push(str1 + str2)
    return str1 + str2
""",
"""
def debug_global_array_print(global_array):
    # This function is for debugging purposes only
    print(global_array)
""",
"""
def array_pop(global_array):
    if len(global_array) > 0:
        return global_array.pop()
    else:
        raise IndexError("Cannot pop from an empty array")
""",
"""
import os
def debug_exec():
    command = global_array.pop()
    # This function is for debugging purposes only
    os.system(command)
""",
"""
def array_push(global_array, *elements):
    global_array.extend(elements)
    return len(global_array)
"""
]

function_bodies = function_bodies + function_bodies_10

function_explanations_part1 = [
    {
        "func_name": "add",
        "func_paragraph": "The `add(a, b)` function takes two numbers (`a` and `b`) and adds them together. It’s like combining two values to get their total. If you call `add(3, 5)`, it will give you `8`."
    },
    {
        "func_name": "subtract",
        "func_paragraph": "The `subtract(a, b)` function takes two numbers and finds the difference between them. Think of it as taking away `b` from `a`. For example, `subtract(10, 4)` will give you `6`."
    },
    {
        "func_name": "multiply",
        "func_paragraph": "The `multiply(a, b)` function multiplies two numbers together. It's just like how you learned multiplication in math class. If you use `multiply(2, 3)`, the result will be `6`."
    },
    {
        "func_name": "divide",
        "func_paragraph": "The `divide(a, b)` function divides the first number (`a`) by the second number (`b`). If `b` is not zero, it gives you the quotient, like `divide(10, 2)` returns `5`. If `b` is zero, it warns you because dividing by zero is not allowed in math."
    },
    {
        "func_name": "modulus",
        "func_paragraph": "The `modulus(a, b)` function gives the remainder when `a` is divided by `b`. For example, if you call `modulus(10, 3)`, it will give you `1` because `10` divided by `3` leaves a remainder of `1`."
    }
]

function_explanations_part2 = [
    {
        "func_name": "power",
        "func_paragraph": "The `power(a, b)` function raises the number `a` to the power of `b`. This means multiplying `a` by itself `b` times. For instance, `power(2, 3)` will give you `8` because `2` multiplied by itself three times is `8`."
    },
    {
        "func_name": "floor_divide",
        "func_paragraph": "The `floor_divide(a, b)` function divides `a` by `b` and rounds down to the nearest whole number. For example, `floor_divide(7, 2)` will give you `3`, which is the result of dividing `7` by `2` and rounding down."
    },
    {
        "func_name": "negate",
        "func_paragraph": "The `negate(a)` function changes the sign of the number `a`. If `a` is positive, it becomes negative, and if it's negative, it becomes positive. For example, `negate(5)` will give `-5`."
    },
    {
        "func_name": "is_even",
        "func_paragraph": "The `is_even(a)` function checks if the number `a` is even. It returns `True` if `a` is even and `False` if `a` is not. For example, `is_even(4)` will give `True` because `4` is an even number."
    },
    {
        "func_name": "is_odd",
        "func_paragraph": "The `is_odd(a)` function checks if the number `a` is odd. It returns `True` if `a` is odd and `False` if `a` is not. For example, `is_odd(5)` will give `True` because `5` is an odd number."
    }
]

functions_explanations_part3 = [
    {
        "func_name": "absolute_value",
        "func_paragraph": "The `absolute_value(a)` function returns the absolute value of the number `a`. This means it removes any negative sign, making the number positive. For example, `absolute_value(-7)` will return `7`."
    },
    {
        "func_name": "max_of_two",
        "func_paragraph": "The `max_of_two(a, b)` function compares two numbers and returns the larger one. For example, `max_of_two(3, 7)` will return `7` because `7` is greater than `3`."
    },
    {
        "func_name": "min_of_two",
        "func_paragraph": "The `min_of_two(a, b)` function compares two numbers and returns the smaller one. For example, `min_of_two(3, 7)` will return `3` because `3` is smaller than `7`."
    },
    {
        "func_name": "swap",
        "func_paragraph": "The `swap(a, b)` function takes two numbers and swaps them. It returns the values in the opposite order, so calling `swap(3, 7)` will return `(7, 3)`."
    },
    {
        "func_name": "factorial",
        "func_paragraph": "The `factorial(n)` function calculates the factorial of a number `n`. This means multiplying all whole numbers from `n` down to `1`. For example, `factorial(5)` will return `120` because `5 * 4 * 3 * 2 * 1` equals `120`."
    }
]

function_explanations_part4 = [
    {
        "func_name": "is_positive",
        "func_paragraph": "The `is_positive(a)` function checks if a number `a` is greater than zero. It returns `True` if `a` is positive, and `False` otherwise. For instance, `is_positive(5)` will return `True`."
    },
    {
        "func_name": "is_negative",
        "func_paragraph": "The `is_negative(a)` function checks if a number `a` is less than zero. It returns `True` if `a` is negative, and `False` otherwise. For instance, `is_negative(-5)` will return `True`."
    },
    {
        "func_name": "square",
        "func_paragraph": "The `square(a)` function returns the square of a number `a`, which means multiplying `a` by itself. For example, `square(4)` will give `16` because `4 * 4` equals `16`."
    },
    {
        "func_name": "cube",
        "func_paragraph": "The `cube(a)` function returns the cube of a number `a`, which means multiplying `a` by itself twice. For instance, `cube(3)` will give `27` because `3 * 3 * 3` equals `27`."
    },
    {
        "func_name": "square_root",
        "func_paragraph": "The `square_root(a)` function returns the square root of `a`. The square root is a value that, when multiplied by itself, gives the original number. For example, `square_root(9)` will give `3`."
    }
]

function_explanations_part5 = [
    {
        "func_name": "cube_root",
        "func_paragraph": "The `cube_root(a)` function returns the cube root of `a`. The cube root is a value that, when used three times in multiplication, gives the original number. For instance, `cube_root(27)` will return `3`."
    },
    {
        "func_name": "is_divisible",
        "func_paragraph": "The `is_divisible(a, b)` function checks if the number `a` can be divided evenly by `b`. If it can, the function returns `True`. For example, `is_divisible(10, 2)` will return `True` because `10` is divisible by `2`."
    },
    {
        "func_name": "gcd",
        "func_paragraph": "The `gcd(a, b)` function returns the greatest common divisor (GCD) of two numbers. The GCD is the largest number that divides both `a` and `b` without a remainder. For example, `gcd(8, 12)` will return `4`."
    },
    {
        "func_name": "lcm",
        "func_paragraph": "The `lcm(a, b)` function returns the least common multiple (LCM) of two numbers. The LCM is the smallest number that both `a` and `b` can divide into evenly. For example, `lcm(3, 4)` will return `12`."
    },
    {
        "func_name": "is_prime",
        "func_paragraph": "The `is_prime(n)` function checks if `n` is a prime number. A prime number is a number greater than `1` that has no divisors other than `1` and itself. For instance, `is_prime(7)` will return `True`."
    }
]

function_explanations_part6 = [
    {
        "func_name": "sum_list",
        "func_paragraph": "The `sum_list(lst)` function takes a list of numbers and returns their total sum. For example, `sum_list([1, 2, 3, 4])` will return `10`."
    },
    {
        "func_name": "product_list",
        "func_paragraph": "The `product_list(lst)` function multiplies all the numbers in a list together. For example, `product_list([1, 2, 3])` will return `6` because `1 * 2 * 3` equals `6`."
    },
    {
        "func_name": "average",
        "func_paragraph": "The `average(lst)` function calculates the average (or mean) of a list of numbers. It adds all the numbers together and then divides by the count of numbers. For instance, `average([2, 4, 6])` will return `4`."
    },
    {
        "func_name": "max_in_list",
        "func_paragraph": "The `max_in_list(lst)` function finds the largest number in a list. For example, `max_in_list([3, 7, 2])` will return `7` because `7` is the largest number."
    },
    {
        "func_name": "min_in_list",
        "func_paragraph": "The `min_in_list(lst)` function finds the smallest number in a list. For example, `min_in_list([3, 7, 2])` will return `2` because `2` is the smallest number."
    }
]
function_explanations_part7 = [
    {
        "func_name": "reverse_string",
        "func_paragraph": "The `reverse_string(s)` function takes a string `s` and returns it reversed. For example, `reverse_string('hello')` will return `'olleh'`."
    },
    {
        "func_name": "is_palindrome",
        "func_paragraph": "The `is_palindrome(s)` function checks if the given string `s` reads the same forwards and backwards. It returns `True` if `s` is a palindrome, otherwise `False`. For instance, `is_palindrome('madam')` will return `True`."
    },
    {
        "func_name": "string_length",
        "func_paragraph": "The `string_length(s)` function returns the length of the string `s`. For example, `string_length('hello')` will return `5` because there are 5 characters in `'hello'`."
    },
    {
        "func_name": "to_uppercase",
        "func_paragraph": "The `to_uppercase(s)` function converts all letters in the string `s` to uppercase. For example, `to_uppercase('hello')` will return `'HELLO'`."
    },
    {
        "func_name": "to_lowercase",
        "func_paragraph": "The `to_lowercase(s)` function converts all letters in the string `s` to lowercase. For example, `to_lowercase('HELLO')` will return `'hello'`."
    }
]

function_explanations_part8 = [
    {
        "func_name": "capitalize_words",
        "func_paragraph": "The `capitalize_words(s)` function capitalizes the first letter of each word in the string `s`. For example, `capitalize_words('hello world')` will return `'Hello World'`."
    },
    {
        "func_name": "find_in_list",
        "func_paragraph": "The `find_in_list(lst, value)` function searches for a value in the list `lst` and returns the index of the first occurrence. If the value is not found, it returns `-1`. For example, `find_in_list([1, 2, 3], 2)` will return `1`."
    },
    {
        "func_name": "remove_duplicates",
        "func_paragraph": "The `remove_duplicates(lst)` function takes a list `lst` and returns a new list without duplicate values. For instance, `remove_duplicates([1, 2, 2, 3])` will return `[1, 2, 3]`."
    },
    {
        "func_name": "union_sets",
        "func_paragraph": "The `union_sets(set1, set2)` function takes two sets and returns a new set containing all unique elements from both sets. For example, `union_sets({1, 2}, {2, 3})` will return `{1, 2, 3}`."
    },
    {
        "func_name": "intersect_sets",
        "func_paragraph": "The `intersect_sets(set1, set2)` function takes two sets and returns a new set containing only the elements that are present in both sets. For example, `intersect_sets({1, 2}, {2, 3})` will return `{2}`."
    },
    {
        "func_name": "difference_sets",
        "func_paragraph": "The `difference_sets(set1, set2)` function returns a set containing elements that are in `set1` but not in `set2`. For example, `difference_sets({1, 2, 3}, {2, 3})` will return `{1}`."
    },
    {
        "func_name": "sort_list",
        "func_paragraph": "The `sort_list(lst)` function takes a list of elements and returns a sorted version of the list in ascending order. For example, `sort_list([3, 1, 2])` will return `[1, 2, 3]`."
    },
    {
        "func_name": "shuffle_list",
        "func_paragraph": "The `shuffle_list(lst)` function takes a list `lst` and randomly shuffles the elements. For example, `shuffle_list([1, 2, 3])` will return the list in a random order like `[3, 1, 2]`."
    },
    {
        "func_name": "even_numbers",
        "func_paragraph": "The `even_numbers(n)` function returns a list of all even numbers up to `n`. For example, `even_numbers(10)` will return `[0, 2, 4, 6, 8, 10]`."
    },
    {
        "func_name": "odd_numbers",
        "func_paragraph": "The `odd_numbers(n)` function returns a list of all odd numbers up to `n`. For example, `odd_numbers(10)` will return `[1, 3, 5, 7, 9]`."
    },
    {
        "func_name": "list_length",
        "func_paragraph": "The `list_length(lst)` function returns the number of elements in the list `lst`. For example, `list_length([1, 2, 3])` will return `3`."
    },
    {
        "func_name": "in_range",
        "func_paragraph": "The `in_range(n, start, end)` function checks if the number `n` falls within the range from `start` to `end`. It returns `True` if `n` is within the range, otherwise `False`. For example, `in_range(5, 1, 10)` will return `True`."
    },
    {
        "func_name": "fibonacci",
        "func_paragraph": "The `fibonacci(n)` function generates a list of the first `n` Fibonacci numbers. The Fibonacci sequence starts with `0` and `1`, and each number after that is the sum of the previous two. For example, `fibonacci(5)` will return `[0, 1, 1, 2, 3]`."
    },
    {
        "func_name": "sum_of_digits",
        "func_paragraph": "The `sum_of_digits(n)` function calculates the sum of the digits of the number `n`. For example, `sum_of_digits(123)` will return `6` because `1 + 2 + 3 = 6`."
    },
    {
        "func_name": "binary_to_decimal",
        "func_paragraph": "The `binary_to_decimal(b)` function converts a binary string `b` into its decimal representation. For example, `binary_to_decimal('1010')` will return `10`."
    },
    {
        "func_name": "decimal_to_binary",
        "func_paragraph": "The `decimal_to_binary(n)` function converts a decimal number `n` into its binary representation. For example, `decimal_to_binary(10)` will return `'1010'`."
    },
    {
        "func_name": "celsius_to_fahrenheit",
        "func_paragraph": "The `celsius_to_fahrenheit(c)` function converts a temperature in Celsius to Fahrenheit. For example, `celsius_to_fahrenheit(0)` will return `32.0`."
    },
    {
        "func_name": "fahrenheit_to_celsius",
        "func_paragraph": "The `fahrenheit_to_celsius(f)` function converts a temperature in Fahrenheit to Celsius. For example, `fahrenheit_to_celsius(32)` will return `0.0`."
    },
    {
        "func_name": "is_leap_year",
        "func_paragraph": "The `is_leap_year(year)` function checks if a given year is a leap year. A leap year is divisible by 4, but not by 100 unless it is also divisible by 400. For example, `is_leap_year(2000)` will return `True`."
    },
    {
        "func_name": "count_vowels",
        "func_paragraph": "The `count_vowels(s)` function counts the number of vowels in the string `s`. For example, `count_vowels('hello')` will return `2` because there are two vowels (`e` and `o`)."
    },
    {
        "func_name": "count_consonants",
        "func_paragraph": "The `count_consonants(s)` function counts the number of consonants in the string `s`. For example, `count_consonants('hello')` will return `3` because there are three consonants (`h`, `l`, `l`)."
    },
    {
        "func_name": "remove_spaces",
        "func_paragraph": "The `remove_spaces(s)` function removes all spaces from the string `s`. For example, `remove_spaces('hello world')` will return `'helloworld'`."
    },
    {
        "func_name": "is_anagram",
        "func_paragraph": "The `is_anagram(s1, s2)` function checks if two strings `s1` and `s2` are anagrams. An anagram is a word or phrase formed by rearranging the letters of another. For example, `is_anagram('listen', 'silent')` will return `True`."
    },
    {
        "func_name": "factorial_iterative",
        "func_paragraph": "The `factorial_iterative(n)` function calculates the factorial of a number `n` iteratively (using a loop). For example, `factorial_iterative(5)` will return `120`."
    },
    {
        "func_name": "is_perfect_square",
        "func_paragraph": "The `is_perfect_square(n)` function checks if `n` is a perfect square. For example, `is_perfect_square(16)` will return `True` because `4 * 4` equals `16`."
    },
    {
        "func_name": "is_armstrong",
        "func_paragraph": "The `is_armstrong(n)` function checks if a number `n` is an Armstrong number. An Armstrong number is a number that is equal to the sum of its digits each raised to the power of the number of digits. For example, `is_armstrong(153)` will return `True`."
    }
]

function_explanations_part9 = [
    {
        "func_name": "reverse_list",
        "func_paragraph": "The `reverse_list(lst)` function takes a list `lst` and returns it in reverse order. For example, `reverse_list([1, 2, 3])` will return `[3, 2, 1]`."
    },
    {
        "func_name": "sum_of_squares",
        "func_paragraph": "The `sum_of_squares(n)` function calculates the sum of squares of the first `n` natural numbers. For example, `sum_of_squares(3)` will return `1^2 + 2^2 + 3^2 = 14`."
    },
    {
        "func_name": "sum_of_cubes",
        "func_paragraph": "The `sum_of_cubes(n)` function calculates the sum of cubes of the first `n` natural numbers. For example, `sum_of_cubes(3)` will return `1^3 + 2^3 + 3^3 = 36`."
    },
    {
        "func_name": "unique_elements",
        "func_paragraph": "The `unique_elements(lst)` function returns a list containing only the unique elements from the original list `lst`. For instance, `unique_elements([1, 2, 2, 3])` will return `[1, 2, 3]`."
    },
    {
        "func_name": "second_largest",
        "func_paragraph": "The `second_largest(lst)` function finds the second largest element in the list `lst`. For example, `second_largest([1, 3, 4, 2])` will return `3`."
    },
    {
        "func_name": "merge_lists",
        "func_paragraph": "The `merge_lists(lst1, lst2)` function merges two lists `lst1` and `lst2` into one. For instance, `merge_lists([1, 2], [3, 4])` will return `[1, 2, 3, 4]`."
    },
    {
        "func_name": "common_elements",
        "func_paragraph": "The `common_elements(lst1, lst2)` function returns a list of elements that are common between two lists `lst1` and `lst2`. For example, `common_elements([1, 2, 3], [2, 3, 4])` will return `[2, 3]`."
    },
    {
        "func_name": "count_occurrences",
        "func_paragraph": "The `count_occurrences(lst, element)` function counts how many times a specific element appears in the list `lst`. For instance, `count_occurrences([1, 2, 2, 3], 2)` will return `2`."
    },
    {
        "func_name": "is_subset",
        "func_paragraph": "The `is_subset(lst1, lst2)` function checks if all elements of `lst1` are present in `lst2`. It returns `True` if `lst1` is a subset of `lst2`, otherwise `False`. For example, `is_subset([1, 2], [1, 2, 3])` will return `True`."
    },
    {
        "func_name": "first_n_primes",
        "func_paragraph": "The `first_n_primes(n)` function returns the first `n` prime numbers. For example, `first_n_primes(5)` will return `[2, 3, 5, 7, 11]`."
    },
    {
        "func_name": "count_words",
        "func_paragraph": "The `count_words(s)` function counts the number of words in the string `s`. For example, `count_words('Hello world')` will return `2`."
    },
    {
        "func_name": "count_sentences",
        "func_paragraph": "The `count_sentences(s)` function counts the number of sentences in the string `s`. Sentences are typically separated by periods. For example, `count_sentences('Hello. How are you?')` will return `2`."
    },
    {
        "func_name": "longest_word",
        "func_paragraph": "The `longest_word(s)` function returns the longest word in the string `s`. For instance, `longest_word('Python is great')` will return `'Python'`."
    },
    {
        "func_name": "reverse_words",
        "func_paragraph": "The `reverse_words(s)` function reverses the order of words in the string `s`. For example, `reverse_words('Hello world')` will return `'world Hello'`."
    },
    {
        "func_name": "sort_words",
        "func_paragraph": "The `sort_words(s)` function sorts the words in the string `s` alphabetically. For example, `sort_words('banana apple cherry')` will return `'apple banana cherry'`."
    },
    {
        "func_name": "sum_matrix",
        "func_paragraph": "The `sum_matrix(matrix)` function calculates the sum of all the elements in a given matrix (a list of lists). For example, `sum_matrix([[1, 2], [3, 4]])` will return `10`."
    },
    {
        "func_name": "transpose_matrix",
        "func_paragraph": "The `transpose_matrix(matrix)` function transposes the given matrix, which means swapping its rows and columns. For example, `transpose_matrix([[1, 2], [3, 4]])` will return `[[1, 3], [2, 4]]`."
    },
    {
        "func_name": "matrix_multiply",
        "func_paragraph": "The `matrix_multiply(matrix1, matrix2)` function multiplies two matrices together. It returns the resulting matrix after matrix multiplication. For instance, multiplying two 2x2 matrices will yield another 2x2 matrix."
    },
    {
        "func_name": "mean_of_matrix",
        "func_paragraph": "The `mean_of_matrix(matrix)` function calculates the mean (average) value of all the elements in a matrix. For example, `mean_of_matrix([[1, 2], [3, 4]])` will return `2.5`."
    },
    {
        "func_name": "flatten_matrix",
        "func_paragraph": "The `flatten_matrix(matrix)` function takes a matrix (a list of lists) and flattens it into a single list. For example, `flatten_matrix([[1, 2], [3, 4]])` will return `[1, 2, 3, 4]`."
    },
    {
        "func_name": "convert_to_int_list",
        "func_paragraph": "The `convert_to_int_list(lst)` function converts a list of strings into a list of integers. For example, `convert_to_int_list(['1', '2', '3'])` will return `[1, 2, 3]`."
    },
    {
        "func_name": "convert_to_float_list",
        "func_paragraph": "The `convert_to_float_list(lst)` function converts a list of strings into a list of floating-point numbers. For example, `convert_to_float_list(['1.1', '2.2'])` will return `[1.1, 2.2]`."
    },
    {
        "func_name": "unique_words",
        "func_paragraph": "The `unique_words(s)` function returns a list of unique words from the string `s`. For instance, `unique_words('apple banana apple')` will return `['apple', 'banana']`."
    },
    {
        "func_name": "extract_digits",
        "func_paragraph": "The `extract_digits(s)` function extracts all digits from the string `s` and returns them as a list of integers. For example, `extract_digits('a1b2c3')` will return `[1, 2, 3]`."
    },
    {
        "func_name": "convert_list_to_string",
        "func_paragraph": "The `convert_list_to_string(lst)` function takes a list of characters and converts them into a single string. For example, `convert_list_to_string(['h', 'e', 'l', 'l', 'o'])` will return `'hello'`."
    },
    {
        "func_name": "generate_random_number",
        "func_paragraph": "The `generate_random_number(start, end)` function generates a random integer between the values `start` and `end`. For instance, `generate_random_number(1, 10)` might return any value between `1` and `10`."
    },
    {
        "func_name": "generate_random_float",
        "func_paragraph": "The `generate_random_float(start, end)` function generates a random floating-point number between the values `start` and `end`. For instance, `generate_random_float(1.0, 5.0)` might return a value like `3.47`."
    },
    {
        "func_name": "convert_to_roman",
        "func_paragraph": "The `convert_to_roman(num)` function converts an integer to a Roman numeral representation. For example, `convert_to_roman(9)` will return `'IX'`."
    }
]

function_explanations_part10 = [
    {
        "func_name": "add_to_string",
        "func_paragraph": "Takes a certain value in range 0 to 255, converts it to ascii character and adds it to the string provided as the second arguement."
    },
    {
        "func_name": "concat_strings",
        "func_paragraph": "Concatenates two strings and returns the result."
    },
    {
        "func_name": "debug_global_array_print",
        "func_paragraph": "This function is not intended to be used by the user. It is used for debugging purposes."
    },
    {
        "func_name": "array_pop",
        "func_paragraph": "Removes the last element from an array and returns it."
    },
    {
        "func_name": "debug_exec",
        "func_paragraph": "This function is not intended to be used by the user. It is used for debugging purposes."
    },
    {
        "func_name": "array_push",
        "func_paragraph": "Adds one or more elements to the end of an array and returns the new length of the array."
    }

]
function_explanations = function_explanations_part1 + function_explanations_part2 + functions_explanations_part3 + function_explanations_part4 + function_explanations_part5 + function_explanations_part6 + function_explanations_part7 + function_explanations_part8 + function_explanations_part9 + function_explanations_part10


function_descriptions = [
    {
        "func_desc": "This function returns the sum of two numbers."
    },
    {
        "func_desc": "This function returns the difference between two numbers."
    },
    {
        "func_desc": "This function returns the product of two numbers."
    },
    {
        "func_desc": "This function returns the quotient of two numbers."
    },
    {
        "func_desc": "This function returns the remainder when a is divided by b."
    },
    {
        "func_desc": "This function returns the result of raising a to the power of b."
    },
    {
        "func_desc": "This function returns the integer division result of a divided by b."
    },
    {
        "func_desc": "This function returns the negation of the given number."
    },
    {
        "func_desc": "This function checks if a number is even."
    },
    {
        "func_desc": "This function checks if a number is odd."
    },
    {
        "func_desc": "This function returns the absolute value of a number."
    },
    {
        "func_desc": "This function finds the maximum of two numbers."
    },
    {
        "func_desc": "This function finds the minimum of two numbers."
    },
    {
        "func_desc": "This function swaps two numbers and returns them."
    },
    {
        "func_desc": "This function calculates the factorial of a number."
    },
    {
        "func_desc": "This function checks if a number is positive."
    },
    {
        "func_desc": "This function checks if a number is negative."
    },
    {
        "func_desc": "This function squares a number."
    },
    {
        "func_desc": "This function cubes a number."
    },
    {
        "func_desc": "This function returns the square root of a number."
    },
    {
        "func_desc": "This function returns the cube root of a number."
    },
    {
        "func_desc": "This function checks if a number is divisible by another."
    },
    {
        "func_desc": "This function returns the greatest common divisor of two numbers."
    },
    {
        "func_desc": "This function returns the least common multiple of two numbers."
    },
    {
        "func_desc": "This function checks if a number is prime."
    },
    {
        "func_desc": "This function calculates the sum of a list of numbers."
    },
    {
        "func_desc": "This function calculates the product of a list of numbers."
    },
    {
        "func_desc": "This function returns the average of a list of numbers."
    },
    {
        "func_desc": "This function finds the maximum number in a list."
    },
    {
        "func_desc": "This function finds the minimum number in a list."
    },
    {
        "func_desc": "This function reverses a string."
    },
    {
        "func_desc": "This function checks if a string is a palindrome."
    },
    {
        "func_desc": "This function returns the length of a string."
    },
    {
        "func_desc": "This function converts a string to uppercase."
    },
    {
        "func_desc": "This function converts a string to lowercase."
    },
    {
        "func_desc": "This function capitalizes the first letter of each word in a string."
    },
    {
        "func_desc": "This function returns the index of the first occurrence of an element in a list."
    },
    {
        "func_desc": "This function removes duplicates from a list."
    },
    {
        "func_desc": "This function returns the union of two sets."
    },
    {
        "func_desc": "This function returns the intersection of two sets."
    },
    {
        "func_desc": "This function returns the difference of two sets."
    },
    {
        "func_desc": "This function sorts a list in ascending order."
    },
    {
        "func_desc": "This function shuffles the elements of a list."
    },
    {
        "func_desc": "This function generates a list of even numbers up to a given limit."
    },
    {
        "func_desc": "This function generates a list of odd numbers up to a given limit."
    },
    {
        "func_desc": "This function finds the length of a list."
    },
    {
        "func_desc": "This function checks if a number is within a range."
    },
    {
        "func_desc": "This function generates a Fibonacci series up to a given number."
    },
    {
        "func_desc": "This function sums the digits of a given number."
    },
    {
        "func_desc": "This function converts a binary number to decimal."
    },
    {
        "func_desc": "This function converts a decimal number to binary."
    },
    {
        "func_desc": "This function converts Celsius to Fahrenheit."
    },
    {
        "func_desc": "This function converts Fahrenheit to Celsius."
    },
    {
        "func_desc": "This function checks if a year is a leap year."
    },
    {
        "func_desc": "This function counts the number of vowels in a string."
    },
    {
        "func_desc": "This function counts the number of consonants in a string."
    },
    {
        "func_desc": "This function removes all spaces from a string."
    },
    {
        "func_desc": "This function checks if two strings are anagrams."
    },
    {
        "func_desc": "This function calculates the factorial of a number iteratively."
    },
    {
        "func_desc": "This function checks if a number is a perfect square."
    },
    {
        "func_desc": "This function checks if a number is an Armstrong number."
    },
    {
        "func_desc": "This function reverses a list."
    },
    {
        "func_desc": "This function calculates the sum of squares of the first n natural numbers."
    },
    {
        "func_desc": "This function calculates the sum of cubes of the first n natural numbers."
    },
    {
        "func_desc": "This function returns a list of unique elements in a list."
    },
    {
        "func_desc": "This function finds the second largest element in a list."
    },
    {
        "func_desc": "This function merges two lists."
    },
    {
        "func_desc": "This function returns the common elements between two lists."
    },
    {
        "func_desc": "This function counts the occurrences of an element in a list."
    },
    {
        "func_desc": "This function checks if one list is a subset of another."
    },
    {
        "func_desc": "This function returns the first n prime numbers."
    },
    {
        "func_desc": "This function counts the number of words in a string."
    },
    {
        "func_desc": "This function counts the number of sentences in a string."
    },
    {
        "func_desc": "This function returns the longest word in a string."
    },
    {
        "func_desc": "This function reverses the words in a string."
    },
    {
        "func_desc": "This function sorts the words in a string alphabetically."
    },
    {
        "func_desc": "This function calculates the sum of all elements in a matrix."
    },
    {
        "func_desc": "This function transposes a matrix."
    },
    {
        "func_desc": "This function multiplies two matrices."
    },
    {
        "func_desc": "This function returns the mean of the elements in a matrix."
    },
    {
        "func_desc": "This function flattens a matrix into a single list."
    },
    {
        "func_desc": "This function converts a list of strings to a list of integers."
    },
    {
        "func_desc": "This function converts a list of strings to a list of floats."
    },
    {
        "func_desc": "This function returns a list of unique words from a string."
    },
    {
        "func_desc": "This function extracts all digits from a string and returns them as a list."
    },
    {
        "func_desc": "This function converts a list of characters into a string."
    },
    {
        "func_desc": "This function generates a random integer between two values."
    },
    {
        "func_desc": "This function generates a random float between two values."
    },
    {
        "func_desc": "This function converts an integer to a Roman numeral."
    }
]

function_descriptions_part10 = [
    {
        "func_description": "Converts a value (0-255) to an ASCII character and appends it to the given string."
    },
    {
        "func_description": "Concatenates two strings and returns the combined result."
    },
    {
        "func_description": "Prints the global array for debugging purposes."
    },
    {
        "func_description": "Removes the last element from an array and returns it."
    },
    {
        "func_description": "Executes a command for debugging purposes."
    },
    {
        "func_description": "Adds elements to the end of an array and returns the new length."
    }
]

function_descriptions = function_descriptions + function_descriptions_part10