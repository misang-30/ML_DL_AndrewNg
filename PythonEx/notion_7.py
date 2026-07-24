# ---------------<PL07-List, for 반복문>---------------
# >>>1. list
# 파이썬에서 리스트는 쉼표로 구분된 표현식들을 대괄호 [ ] 안에 넣으면 생성됩니다.
# ex) [1, "two", 6 / 2] 는 리스트입니다. 이 리스트에는 세 개의 요소가 있습니다
# 리스트는 같은 종류의 데이터만 포함 -> 동질(Homogeneous)
# 다른 종류의 데이터도 포함 -> 이질(Heterogeneous)

x = [1,"two", 6/2]
def f(n):
    return [n,2*n, 3*n]

f(2)
# [2,4,6] 출력

f('yo')
# ['yo','yoyo','yoyoyo']


# 파이썬에서 리스트는 list라는 클래스로 정의되어 있습니다.
# 따라서 리스트 객체(리스트 인스턴스)는 여러 **메서드(method)**와 **속성(attribute)**을 가집니다.

# 이 메서드와 속성을 확인하려면 dir() 함수를 사용할 수 있습니다.
# 리스트는 연산자 오버로딩을 지원합니다.
# 특정 연산자를 리스트에 맞게 “재정의”해서 사용할 수 있다는 뜻입니다.
[1, 2] + [3, 4]  # 결과: [1, 2, 3, 4]
[1, 2] * 3  # 결과: [1, 2, 1, 2, 1, 2]

# append, sort, extend 등이 있다.


w.append('quartet') # Append value to list.
w # 결과: ['a', 'bee', 'duo', 'sea', 'solo', 'trio', 'quartet']

w.sort()           # Sort list again.
w # 결과: ['a', 'bee', 'duo', 'quartet', 'sea', 'solo', 'trio']

z = []             # Create an empty list.
print(z)
# 결과: []

len(z)
# 결과: 0

z.append("first")  # Append value to empty list.
z
# 결과: ['first']

# Extend z with the values from another list.
z.extend(["second", "third", "fourth"])
z
# 결과: ['first', 'second', 'third', 'fourth']

w = ['a', 'b', 'c']
w.append(['d', 'e'])
w
# 결과: ['a', 'b', 'c', ['d', 'e']]

u = ['a', 'b', 'c']
u.extend(['d', 'e'])
u
# 결과: ['a', 'b', 'c', 'd', 'e']

# Concatenate two lists.
w = ['a', 'bee', 'sea'] + ["solo", "duo", "trio"]
w
# 결과: ['a', 'bee', 'sea', 'solo', 'duo', 'trio']

len(w)           # Determine length.
# 결과: 6

dir(w)           # Show methods and attributes.
# 결과: 리스트 메서드 목록 출력

type(w.sort)
# 결과: <class 'builtin_function_or_method'>

w.sort()         # Sort list.
print(w)
# 결과: ['a', 'bee', 'duo', 'sea', 'solo', 'trio']


# ---------------------------------------------------------------------------------
# >>>2. for
"""
for <item> in <iterable>:
    <body>
"""
# <item> (루프 변수): 반복문이 실행될 때마다 리스트(혹은 이터러블) 내의 
# 개별 요소들이 차례대로 할당되는 변수입니다.
# in: 루프 변수와 반복할 대상을 연결해 주는 키워드입니다.
# <iterable> (이터러블): 반복 가능한 데이터 타입을 의미합니다. (예: 리스트, 문자열 등)
# <body> (본문): 각 요소에 대해 실행할 실제 코드 블록입니다.

# * 루프 변수 (Loop Variable)
# <item>은 매 반복마다 새로운 값을 담는 변수라고 생각하면 됩니다.
# 기술적으로는 lvalue(값을 할당받을 수 있는 위치) 역할을 하며, 리스트의 첫 번째 요소부터 마지막 요소까지 순서대로 이 변수에 담기게 됩니다.

# * 이터러블 (Iterable)
# 정의: 요소들을 한 번에 하나씩 따로 반환할 수 있는 데이터 타입을 말합니다.
# 종류: 파이썬에서 **리스트(List)**는 가장 대표적인 이터러블의 한 종류입니다.


# ---------------------------------------------------------------------------------
# >>>3. indexing

"""
>>> xlist = ["Do", "Re", "Mi", "Fa", "So", "La", "Ti"]
>>> xlist[0]                 # 첫 번째 요소 (First element)
'Do'
>>> xlist[1]                 # 두 번째 요소 (Second element)
'Re'
>>> xlist[1 + 1]             # 세 번째 요소 (Third element)
'Mi'
>>> len(xlist)               # 리스트의 길이 (Length of list)
7
>>> xlist[len(xlist) - 1]    # 마지막 요소 (Last element)
'Ti'
>>> xlist[len(xlist)]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: list index out of range

>>> ["Fee", "Fi", "Fo", "Fum"][1]
'Fi'
>>> ["Fee", "Fi", "Fo", "Fum"][3]
'Fum'
>>> ([1, 2, 3] + ['a', 'b', 'c'])[4]
'b'

"""



# ---------------------------------------------------------------------------------
# >>>4. range()
# 파이썬은 **range()**라는 함수를 제공하며, 
# 이 함수는 **정수 시퀀스(순서 있는 정수들)**를 생성합니다.
# range() 함수는 실제로 정수들의 리스트를 만들어내지는 않습니다.
# range(start, stop, increment)

list(range(0,5,1))
# [0,1,2,3,4] 출력



# ---------------------------------------------------------------------------------
# >>>5. Mutablity(변경 가능), Immutability, Tuple


# 튜플(tuple)은 또 다른 데이터 타입입니다. 동작 방식은 리스트와 꽤 비슷합니다.
# 튜플의 요소에 접근할 때도 대괄호 [ ] 안에 인덱스를 사용합니다.
# 튜플의 길이는 len() 함수로 구할 수 있습니다.

# 리스트와 튜플의 한 가지 차이점은, 튜플은 괄호 ( ) 안에 쉼표로 구분된 
# 데이터를 넣어 생성한다는 것입니다. (리스트는 대괄호 [ ] 사용)

# 리스트와 튜플의 가장 큰 차이점은 튜플은 변경 불가능(immutable)하다는 것입니다. 
# 즉, 한 번 만든 튜플의 값은 바꿀 수 없습니다.
# 튜플의 요소 값을 바꿔야 할 경우, list() 함수를 이용해 튜플을 리스트로 변환하면 됩니다.



# ---------------------------------------------------------------------------------
# >>>6. 리스트를 이용한 동시할당 

# 동시 할당(simultaneous assignment)에서는 등호(=) 오른쪽에 두 개 이상 쉼표로 
# 구분된 표현식이 있고, 왼쪽에도 **같은 수의 쉼표로 구분된 변수(lvalue)**가 있습니다.

# 쉼표로 구분된 표현식들의 모음은 자동으로 튜플(tuple)을 형성합니다. 
# (괄호로 둘러싸여 있든 아니든 상관없습니다.)


# 왼쪽 변수의 개수 = 오른쪽 튜플의 요소 개수
# 예시) a, b, c = 1, 2, 3
# 오른쪽 (1, 2, 3)은 사실 튜플 (1, 2, 3)이고,
# 왼쪽 변수 a, b, c의 개수와 맞아야 동시 할당이 가능하다는 의미입니다.

"""
>>> x, y, z = 1, 2, 3        # Tuple to right, without parentheses.
>>> print(x, y, z)
1 2 3
>>> r, s, t = (1, 2, 3)    # Tuple to right, with parentheses.
>>> print(r, s, t)
1 2 3
>>> a, b, c = [1, 2, 3]    # List to right.
>>> print(a, b, c)
1 2 3
>>> xlist = [1, 2, 3]
>>> i, j, k = xlist
>>> print(i, j, k)
1 2 3

>>> p = ['a', 'b', 'c', 'd']  # Assign a list of strings to p.
>>> n = (12, 24)               # Assign a tuple of integers to n.
>>> print(p, n)                # Print p and n.
['a', 'b', 'c', 'd'] (12, 24)
>>> p, n = n, p                # Simultaneous assignment to swap p and n.
>>> print(p, n)                # See what p and n are now.
(12, 24) ['a', 'b', 'c', 'd']
>>> w = "WoW"                  # Define string.
>>> print(p, n, w)             # Print variables.
(12, 24) ['a', 'b', 'c', 'd'] WoW
>>> p, n, w = w, p, n          # Swap variables.
>>> print(p, n, w)             # Print variables again.
WoW (12, 24) ['a', 'b', 'c', 'd']

"""
# ---------------------------------------------------------------------------------
