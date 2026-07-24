# ---------------<PL10-String >---------------


# 1).String ---------------------------------------
'''
• 문자열 리터럴은 작은따옴표('), 큰따옴표("), 또는 같은 따옴표를 세 번 반복한 형태로 둘러싸서 표현한다.
• 따옴표를 세 번 반복한 경우(\''' 또는 """), 문자열은 여러 줄에 걸쳐 작성할 수 있다.
• 문자열은 불변(immutable) 이다.
• 문자열은 + 연산자를 사용하여 연결(concatenation)할 수 있다.
• 문자열은 정수와 곱하면 반복된다.
• str() 함수는 인자를 문자열로 변환한다.

• 문자열은 시퀀스(sequence) 이다. 따라서 다음과 같은 특성을 가진다:
– 문자열은 for 반복문의 iterable로 사용할 수 있으며, 이 경우 반복 변수는 문자열의 각 문자(character)를 하나씩 갖는다.
– len() 함수는 문자열의 길이, 즉 문자 개수를 반환한다.
– 대괄호([]) 안에 인덱스를 지정하여 문자열의 개별 문자에 접근할 수 있다.
– 슬라이싱(slicing)을 사용하여 문자열의 일부(또는 전체)를 얻을 수 있다.

'''

# ex1).
# 1. 기본적인 문자열 정의 및 출력
s1 = 'This is a string.'
print(s1)
## This is a string.

# 2. 다양한 따옴표를 사용한 문자열 연결 (Concatenation)
s2 = "This" + 'is' + """also""" + '''a string.'''
# s2의 내용 확인
# 'Thisisalsoa\nstring.' 
print(s2)
## Thisisalsoa
## string.

# 3. 역슬래시(\)를 이용해 여러 줄에 걸쳐 작성한 단일 라인 문자열
s3 = "this is a \
single line"
print(s3)
## this is a single line

# 4. 문자열 반복 (String repetition)
s4 = "Why? " * 3
print(s4)
## Why? Why? Why? 

# 5. 문자열 내부의 해시(#) 기호 확인
# 문자열 안에 있는 #은 주석으로 처리되지 않습니다.
s5 = "this is # not a comment"
print(s5)


# ex2).
# 변수 선언 및 초기화
x = 10
y = 23.4
zlist = [1, 'a', [2, 3]]
s0 = "Hi!"

# 각 변수를 문자열(str)로 변환하여 모두 연결
all = str(x) + str(y) + str(zlist) + str(s0)

# 결과 출력
print(all)
## 1023.4[1, 'a', [2, 3]]Hi!


# ex3).
s1 = '"Unbroken" by Laura Hillenbrand'
print(s1)
## "Unbroken" by Laura Hillenbrand

# 문자열 길이 확인
len(s1)  # 32

# 슬라이싱 (16번부터 19번 인덱스까지)
s1[16 : 20] 
## 'aura'

# 인덱싱 (2번째 문자 추출)
s1[1]
## 'U'

# 문자열 수정 시도 (오류 발생: 문자열은 불변 객체임)
# s1[1] = 'u' 
## TypeError: 'str' object does not support item assignment

# 반복문을 이용한 문자 출력 (하이픈으로 연결)
s2 = "Loopy!"
for ch in s2:
    print(ch, end="-")
## L-o-o-p-y-!-

# 음수 인덱스를 활용한 역순 출력 예시
for i in range(len(s2)):
    print(-i, s2[-i])
## 0 L (참고: s2[0]은 L이며, 루프의 첫 번째 i는 0임)
## -1 !
## -2 y
## -3 p
## -4 o
## -5 o


# 2). chr(), ord()  ---------------------------------------
# chr()와 ord()는 문자(character) ↔ 정수(유니코드 코드포인트) 를
#  서로 변환해주는 파이썬 내장 함수야.
# 문자를 숫자로 변환 (ord)
ord('A'), ord('B'), ord('!'), ord(' ')
## (65, 66, 33, 32)

# 숫자를 문자로 변환 (chr)
chr(65), chr(66), chr(33), chr(32)
## ('A', 'B', '!', ' ')

# 'Z'와 90은 ASCII 한 쌍
ord('Z'), chr(90)
## (90, 'Z')

# 중첩 함수: 원래 인수로 돌아감
ord(chr(90))
## 90

chr(ord('Z'))
## 'Z'

# 65부터 74까지의 숫자를 문자로 변환하여 출력
for i in range(65, 75):
    print(chr(i), end="")
## ABCDEFGHIJ

# 문자열의 각 문자를 ASCII 숫자로 변환하여 출력
for ch in "ASCII = numbers":
    print(ord(ch), end=" ")
## 65 83 67 73 73 32 61 32 110 117 109 98 101 114 115


# 3). 다양한 문자열 메서드  ---------------------------------------
# • 문자열 객체는 여러 가지 메서드를 제공한다.
# • 이러한 메서드들은 dir() 함수에 문자열 리터럴, 문자열 변수, 또는 str() 함수(괄호를 사용하든 사용하지 않든)를 인자로 주어 확인할 수 있다.
# • 예를 들어, dir("")와 dir(str)는 문자열에서 사용할 수 있는 모든 메서드를 나열한다.
# • 예를 들어 count() 메서드에 대한 도움말을 보고 싶다면 다음과 같이 입력할 수 있다:
#   help("".count)
# 
# lower(), upper(), capitalize(), title(), swapcase(), 
# count(), strip(), lstrip(), rstrip(), find(), index()
# replace(), split(), join(), 
# __repr__() 메서드
# • 파이썬의 모든 객체는 __repr__() 메서드를 가지고 있다.
# • 이 메서드는 객체의 “공식적인(official) 문자열 표현” 을 제공한다.
# • 즉, __repr__()은 “representation(표현)” 을 의미한다고 생각할 수 있다.
# __repr__()은 객체를 다시 만들어낼 수 있을 정도로 정확한 문자열을 반환하는 것이 목표야.
# 보통 디버깅할 때, 또는 객체를 출력할 때 내부적으로 사용돼.

# 4). format() 메소드  ---------------------------------------
# • 우리는 원하는 모양 그대로의 문자열을 만들어낼 수 있다.
# • 이러한 문자열은 포맷 문자열(format string) 과 format() 메서드를 함께 사용하여 생성한다.
# • 포맷 문자열은 그대로 출력되기를 원하는 문자들(literal characters) 과
# 중괄호 {} 로 둘러싸인 치환 필드(replacement fields) 로 구성된다.
# • 이 예제들에서 치환 필드(replacement fields), 즉 중괄호 {} 는 단순한 플레이스홀더(자리 표시자) 역할을 한다.
# • 이는 format() 메서드의 인자(들)가 들어갈 위치(또는 위치들) 를 표시하는 것이다.
# • (플레이스홀더인) 치환 필드와 format() 메서드의 인자 사이에는 1:1 대응 관계가 있다.


# 특수 기호 그대로 출력하기
print("These {}'s are simply braces.") ## These {}'s are simply braces.

# 기본적인 .format() 사용
print("Hello {}!".format("Jack")) ## Hello Jack!
print("Float: {}, List: {}".format(1 / 7, [1, 2, 3])) 
## Float: 0.14285714285714285, List: [1, 2, 3]

# 인덱스를 활용한 값 배치
print("Hello {0} and {1}. Or, is it {1} and {0}?".format("Jack", "Jill"))
## Hello Jack and Jill.  Or, is it Jill and Jack?

# 같은 인덱스 중복 사용
print("Hello {0} {0} {0} {0}.".format("Major"))
## Hello Major Major Major Major.

# 에러 케이스: 인덱스 범위를 벗어난 경우
# "Hello {} {} {} {}.".format("Major") 
## IndexError: tuple index out of range

### 폭 지정
"{:5}".format(42)  # 5칸 폭을 확보하고 42 출력

### 정렬 지정
"{:<10}".format("left")    # 왼쪽 정렬
"{:^10}".format("center")  # 가운데 정렬
"{:>10}".format("right")   # 오른쪽 정렬
"{:=10}".format(-42)       # 부호 왼쪽, 숫자 오른쪽

### 채움문자, Zero Padding
"{:_<10}".format("hi")   # 왼쪽 정렬, 빈 공간은 '_'로 채움
"{:*>10}".format("hi")   # 오른쪽 정렬, 빈 공간은 '*'로 채움
"{:0>5}".format(42)      # 오른쪽 정렬, 빈 공간은 '0'으로 채움 (zero padding)

### 실수의 정밀도(Precision)
pi = 3.14159265
print("{:.2f}".format(pi))  # 소수점 둘째 자리까지
# 출력: 3.14

print("{:.4f}".format(pi))  # 소수점 넷째 자리까지
# 출력: 3.1416

### 자료형
"""
| 지정자 | 설명                     | 예제                                     |
| --- | ---------------------------| ---------------------------------------- |
| `d` | 정수(10진수)                | `"{:d}".format(42)` → `42`               |
| `b` | 정수(2진수)                 | `"{:b}".format(5)` → `101`               |
| `o` | 정수(8진수)                 | `"{:o}".format(8)` → `10`                |
| `x` | 정수(16진수, 소문자)        | `"{:x}".format(255)` → `ff`              |
| `X` | 정수(16진수, 대문자)        | `"{:X}".format(255)` → `FF`              |
| `f` | 고정 소수점 실수            | `"{:.2f}".format(3.14159)` → `3.14`      |
| `e` | 지수 표기법 소수            | `"{:.2e}".format(1234)` → `1.23e+03`      |
| `E` | 지수 표기법 소수(대문자 E)  | `"{:.2E}".format(1234)` → `1.23E+03`      |
| `g` | 일반 형식(가장 간단한 표현) | `"{:g}".format(0.0001234)` → `0.0001234` |
| `%` | 퍼센트 표시                | `"{:.2%}".format(0.1234)` → `12.34%`     |
| `s` | 문자열                     | `"{:s}".format("Hello")` → `"Hello"`     |


"""

