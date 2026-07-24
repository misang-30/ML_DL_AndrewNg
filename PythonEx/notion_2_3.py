# ---------------<PL02-Introduction>---------------
# 1. print
print("hello", end= " @\n")
print("hello\n")
print(42, "42") # 42 42 출력

# 2. help()
help() # 필요한 명령어 설명

# ---------------<PL03-Core Basics>---------------

# 1. Literals and Types
# string 리터럴은 따옴표에 갖힌 characters의 모음.
# 리터럴은 value를 정확하게 표현하는 것이다.
# 많은 자료형은 type()함수로 결정할 수 있다.
# loat 리터럴은 소수점을 포함하거나, 10의 거듭제곱 지수 표시자인 e를 포함한다.
#    ex) 1.1e2, 11e1, 110.0 은 모두 같은 float 값이다.

# 문자열에서 \n은 **줄바꿈 문자(newline)**를 의미한다.
# 문자열의 문자는 백슬래시(\)를 사용해 이스케이프할 수 있다.
# 이는 문자의 의미를 바꾼다.
# 예를 들어 문자열 안에서 따옴표 앞에 \를 붙이면 문자열 종료가 아니라 문자 자체로 처리된다.

# \' 작은따옴표 문자 자체
# \" 큰따옴표 문자 자체ㄴ
# \\ 백슬래시 자체


7 # int literal
3.14 # float literal
-2.001 # negative sign literal
1.23e3 # 123.0 Exponential literal
1.23e-3 # 0.00123 Exponential literal
1e0 # 1.0 int literal
"Hello" # 쌍따옴표 String
'World' # 따옴표 String

# 2. Expressions, Arithmetic Operator, Precedence
# (표현식, 산술 연산자, 연산자 우선순위 )
''' 
여러 연산이 포함된 표현식은 우선순위 규칙에 따라 평가된다.
**거듭제곱(**)**이 가장 높은 우선순위를 가진다.

곱셈(*), 정수/실수 나눗셈(/, //), 모듈로(%)는 같은 우선순위를 가지며
거듭제곱보다는 낮고 덧셈/뺄셈보다는 높다.

덧셈(+)과 뺄셈(-)은 같은 우선순위를 가진다.

같은 우선순위의 연산은 왼쪽에서 오른쪽으로 평가되며,
거듭제곱만 예외로 오른쪽에서 왼쪽으로 평가된다.
'''



type(42) # <class 'int'>
type("hello world!") # <class 'str'>
type('hello') # <class 'str'>

4-5 # expression
(5+4)*3 # expression with precedence


 
# 3. Statements, Assignment Operator
x=10
x # 10출력
x=x+1 # x는 11


# 4. Cascaded and Simultaneous Assignment (연쇄 대입, 동시 대입)
current = "password1"
old = "password2"
temp=current
current = old
old=temp # current와 old 사이의 데이터 교환

x,y = 10+20, 20-40 # 동시할당


# 5. Multi-Line Statements and Multi-Line Strings
x=(123123123 *12321
   +121314+131
   + 1241
   +6)
s= """strtr 
sqrq
qrqt
"""
# strtr 
# sqrq
# qrqt 출력
a= '''adaf
adfafa
agaga
'''

# adaf
# adfafa
# agaga 출력

# 6. Identifiers and Keywords
# 식별자는 키워드 쓰면 안된다. 이미 있는것이라서
# 식별자 명명 규칙 잘 지켜라


# 7. Name and Namespaces
# 파이썬에서는 네임스페이스 안에 Name(=변수), Object(실제 메모리) 2가지로 구성되어 있다.
# Name과 Object를 바인딩하여 사용한다.
# Object에는 값이 들어가 있다.


# 8. Additional Arithmetic Operators
2 ** 3  # 2  ^3
143 // 25 # 5출력 >> Floor Division 이라한다. 몇 층인지 알려주는 느낌.
143 % 25 # 18 출력 >> Modulo ( 나머지 연산)
divmod( 143, 25) # 143//25, 143%25 연산이다.

x=22
x +=7 # x=x+7
x //=7 # x=x//7 몫 나눗셈이다.
x *=7 # x=x*7
x /=7 # x=x/7 실수 나눗셈이다.


# ---------------<PL04->---------------