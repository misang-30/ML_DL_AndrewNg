# ---------------<PL09-Modules and Import Statements >---------------

#강의 보고 정리하든가
# 1). Modules and import Statements ---------------------------------------
# 

z=111
def f(x) :
    return 2* x

dir()
# 인자 없이 dir()을 실행하면, **현재 네임스페이스(Namespace)**에 
# 있는 모든 이름(변수, 함수, 모듈 등)을 리스트 형태로 보여줍니다.
# 출력 결과인 ['__builtins__', ..., 'f', 'z']를 보면 앞서 정의한 
# f와 z가 포함되어 있음을 알 수 있습니다.

dir(__builtins__)
# 파이썬이 기본적으로 제공하는 내장 함수와 예외 객체들을 확인하는 명령어입니다.

# 리스트에 나온 abs, bin, dict, int, len, print, sum, zip 등은 우리가 별도의 import 없이 바로 사용할 수 있는 기능들입니다.

# 2). Modules 임포트 하기  ---------------------------------------
# math 모듈은 방대한 양의 수학 함수들을 제공합니다.
# 이러한 함수들을 사용하기 위해서는 import 키워드 뒤에 모듈 이름(예: math)을 붙여서 작성합니다.
# 일반적으로 이 문장(import 문)은 파이썬 프로그램이나 코드 파일의 맨 윗부분에 작성합니다.
# 이 명령을 실행하면 math라는 이름의 '모듈 객체'가 생성됩니다

import math
type(math)
# <class 'module'>

dir(math)
# ['__doc__', '__file__', '__name__', '__package__', 'acos', 'acosh', 
# 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'ceil', 'copysign', 
# 'cos', 'cosh', 'degrees', 'e', 'erf', 'erfc', 'exp', 'expm1', 
# 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum', 'gamma', 
# 'hypot', 'isfinite', 'isinf', 'isnan', 'ldexp', 'lgamma', 'log', 
# 'log10', 'log1p', 'modf', 'pi', 'pow', 'radians', 'sin', 'sinh', 
# 'sqrt', 'tan', 'tanh', 'trunc']

help(math.cos)  # Obtain help on math.cos.
# Help on built-in function cos in module math:
# 
# cos(...)
#     cos(x)
# 
#     Return the cosine of x (measured in radians).

math.cos(0)  # Cosine of 0.
# 1.0

math.pi  # math module provides pi = 3.1414...
# 3.141592653589793

math.cos(math.pi)  # Cosine of pi.
# -1.0

cos(0)  # cos() is not defined. Must use math.cos().
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# NameError: name 'cos' is not defined


# 3). Complex Numbers 사용하기  ---------------------------------------
"""
복소수 표현하는 방식.

1.복소평면에서의 표현
-1.직교 좌표 형태 ( z= x+ jy)
실수부 : x
허수부 : y

-2. 극좌표 형태 ( z = r \angle \theta)
크기 : r
위상(Phase,\theta ) : 양의 실수축과 원점에서 점을 잇는 선 사이의 각도.

2.파이썬에서의 복소수
파이썬은 복소수를 기본 데이터 타입(complex)으로 지원한다.
-표기법 : 파이썬에서는 허수 단위로 j를 사용한다.
-z.real : 복소수의 z의 실수부 반환
-z.imag : 복소수의 z의 허수부 반환

파이썬의 cmath 모듈을 사용하면 복소수에 특화된 수학함수를 쉽게 사용가능.



"""
z1 = 1 + 1j  # Create complex number z1.
type(z1)  # Check z1's type.
# <class 'complex'>

z1  # Check z1.
# (1+1j)

j = 10  # Set j to 10.
z2 = 1 + j  # Create integer z2 -- not complex!
z2  # Check z2.
# 11

# z3 = 5.6 - j7.3  # Attempt to create complex number z3.
# SyntaxError: invalid syntax

z3 = 5.6 - 7.3j  # Correct way to create complex number z3.
z1 + z3  # Sum of complex numbers.
# (6.6-6.3j)

z1 * z3  # Product of complex numbers.
# (12.899999999999999-1.7000000000000002j)


# ------------------------
z = 3 + 4j
dir(z)
# ['__abs__', '__add__', '__bool__', '__class__', '__delattr__', '__divmod__', '__doc__', '__eq__', '__float__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__int__', '__le__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__pos__', '__pow__', '__radd__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rmod__', '__rmul__', '__rpow__', '__rsub__', '__rtruediv__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', 'conjugate', 'imag', 'real']

z.real  # The real part of z.
# 3.0

z.imag  # The real imaginary part of z.
# 4.0

type(z.real)
# <class 'float'>

(z.real ** 2 + z.imag ** 2) ** 0.5  # Magnitude of z.
# 5.0

abs(z)  # Simpler way to obtain magnitude.
# 5.0


# -------------------------------
import cmath
dir(cmath)
# ['__doc__', '__file__', '__name__', '__package__', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh', 'e', 'exp', 'isfinite', 'isinf', 'isnan', 'log', 'log10', 'phase', 'pi', 'polar', 'rect', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']

z1 = 1j
cmath.sqrt(z1)
# (0.7071067811865476+0.7071067811865475j)

cmath.cos(5.6 - 7.6j)
# (774.8664714775274-630.6970442971782j)

z2 = 3 + 4j
cmath.polar(z2)
# (5.0, 0.9272952180016122)

# -------------------------------

import math
z1 = 1j  # z1 has zero real part and imaginary part of 1.
z1 * z1  # z1 squared is negative 1.
# (-1+0j)

# Cannot use math.sqrt() on a complex number.
# math.sqrt(z1)
# TypeError: can't convert complex to float

# Cannot use math.sqrt() on a complex number.
# math.sqrt(-1)
# ValueError: math domain error

# Cannot use math.cos() on a complex number.
# math.cos(5.6 - 7.3j)
# TypeError: can't convert complex to float

# 4). import ... as ... 구문  ---------------------------------------

"""
이는 특정 모듈을 불러올 때, 코드 내에서 사용할 이름을 더 짧거나 기억하기 쉬운 **별칭(Alias)**으로 지정하는 방법입니다.

*주요 특징 및 예시
-사용자 정의 식별자: 모듈의 원래 이름 대신 내가 정한 이름을 식별자로 사용할 수 있습니다.
-코드 간결화: 모듈 이름이 길 경우 이를 줄여서 코딩 효율을 높일 수 있습니다.
-충돌 방지: 서로 다른 모듈이 같은 이름의 함수를 가지고 있을 때 별칭을 주어 구분할 수 있습니다.

"""

import math as realmath, cmath as complexmath
realmath.sqrt(2) # Real function, integer argument.
# 1.4142135623730951

complexmath.sqrt(2) # Complex function, integer argument.
# (1.4142135623730951+0j)


# 5). 모듈의 일부분만 importing 하는 것.  ---------------------------------------

"""
# Import a single object.
from <module> import <object>

# Import a single object and assign to an alternative name.
from <module> import <object> as <ident>

# Import multiple objects.
from <module> import <object1>, <object2>, ..., <objectN>

# Import multiple objects and assign to alternative names.
# The as qualifier may be omitted as appropriate.
from <module> import <object1> as <ident1>, ..., <objectN> as <identN>


"""
 
"""  에러 뜨는 경우
from cmath import sqrt
from math import import sqrt
sqrt(-1)
# Traceback (most recent call last):
# File "<stdin>", line 1, in <module>
# ValueError: math domain error

"""

# 동시에 여러 개 임포트
from math import  sqrt as rsqrt, pi
from cmath import sqrt as csqrt, pi as cpi
# (sqrt -> csqrt) , (pi -> cpi)
rsqrt(2)
# 1.4142135623730951

csqrt(2+2j)
# (1.5537739740300374+0.6435942529055826j)

print(pi, cpi, pi - cpi)
# 3.141592653589793 3.141592653589793 0.0





# 6). 모듈의 전부 importing 하는 것. ( * 사용한다) ---------------------------------------


from math import * # Import everything from the math module.
pi
## 3.141592653589793

sqrt(2)
## 1.4142135623730951

# Cosine of pi / 2 is zero. But, because of finite precision, the
# result isn't exactly zero (but it's close!).
cos(pi / 2)
## 6.123233995736766e-17

# 7). 자체 제작 Module 사용 ---------------------------------------

"""
module은 단순한 .py 파일이다.
따라서, 우리는 함수나 다른 오브젝트의 모음을 포함하는 모듈을 쉽게 만들 수 있다. 


"""
# ----------------<my_mod.py>

"""
This module contains two functions and one globally defined 
integer.
"""

scale = 2

def scaler(arg) :
    """Return the scaled value of the argument."""
    return scale * arg

def reverser(xlist) :
    """Return the reverse of the argument."""
    # Reverse using a slice with negative increment and default start
    # and stop.
    return xlist[ : : -1]


# ----------------<main.py>
import my_mod

my_mod.scale
# 2

my_mod.scaler(42.0)
# 84.0

my_mod.scaler("liar")
# 'liarliar'

my_mod.reverser("!pleH")
# 'Help!'

my_mod.scale = 3  # Change value of my_mod.scale.
my_mod.scaler("La")  # See that change affects my_mod.scaler().
# 'LaLaLa'

help(my_mod)  # See what help is available for my_mod.
# Help on module my_mod:
# 
# NAME
#     my_mod
# 
# DESCRIPTION
#     This modules contains two functions and one globally defined 
#     integer.
# 
# FUNCTIONS
#     reverser(xlist)
#         Return the reverse of the argument.
# 
#     scaler(arg)
#         Return the scaled value of the argument.
# 
# DATA
#     scale = 3
# 
# FILE
#     /Users/schneidj/Documents/my_mod.py


# 8). 모듈을 불러올 때 검색 경로를 확인하고 관리하는 방법 ---------------------------------------

"""
1. 현재 작업 디렉토리(CWD) 활용 및 변경
파이썬은 모듈을 찾을 때 가장 먼저 **현재 작업 디렉토리(Current Working Directory)**를 확인합니다.
현재 경로 확인: os 모듈을 사용하여 확인할 수 있습니다

import os
os.getcwd()  # 현재 작업 중인 폴더 경로 표시

작업 경로 변경: 원하는 모듈이 있는 폴더로 작업 디렉토리를 직접 옮길 수 있습니다.
os.chdir("C:/Users/guido/My Documents")
참고: 윈도우 경로를 입력할 때는 슬래시(/)를 쓰거나, 역슬래시를 두 번(\\) 써서 경로를 표시합니다.

"""

"""
2. 모듈 검색 경로(sys.path)에 추가
작업 디렉토리를 매번 바꾸는 것이 번거롭다면, 파이썬이 모듈을 찾는 "리스트(Path)"에 특정 폴더를 추가할 수 있습니다.

방법: sys 모듈의 append 함수를 사용합니다.
import sys
sys.path.append("C:/Users/guido/My Documents")

장점: 현재 작업 디렉토리를 유지하면서도 다른 폴더에 있는 모듈을 자유롭게 가져올 수 있습니다.

"""

"""
3. 표준 라이브러리 경로(site-packages)에 저장

가장 단순한 방법은 파이썬이 기본적으로 모듈을 모아두는 특정 폴더에 파일을 직접 넣는 것입니다.

위치: 윈도우 기준, 보통 site-packages라는 폴더에 모듈을 저장합니다.

특징: 여기에 모듈을 넣어두면 별도의 경로 설정 없이 어디서든 import 명령어로 바로 불러올 수 있어 매우 편리합니다. 
(주로 외부 라이브러리들이 설치되는 위치입니다.)

"""


