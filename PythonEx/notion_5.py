# ---------------<PL05-Function>---------------
# >>>1. Function
type(None) # <class 'NoneType'> 출력
x = print("hello")
type(x) # <class 'NoneType'> 출력
x # 아무것도 출력하지 않는다.
print(x) # None 출력



# void Function
def bmi():
    # Define function
    weight = float(input("Enter weight [pounds]: "))
    height = float(input("Enter height [inches]: "))
    bmi = 703 * weight / (height * height)
    print("Your body mass index is:", bmi)

# Non void Function
def make_change(total):
    """  >> 이 부분은 help(make_change) 하면 나오는 텍스트이다.
    Calculate the change in terms of dollars, quarters,
    dimes, nickels, and pennies in 'total' pennies.
    """
    dollars, remainder = divmod(total, 100)
    quarters, remainder = divmod(remainder, 25)
    dimes, remainder = divmod(remainder, 10)
    nickels, pennies = divmod(remainder, 5)
    return dollars, quarters, dimes, nickels, pennies





# ---------------------------------------------------------------------------------

# >>>2. Optional Parameters

# 구분자(sep) 기본값은 공백
print("Hello", "World") # Hello World 출력

# 구분자를 "-*-"로 설정
print("Hello", "World", sep="-*-") # Hello-*-World 출력

# 세미콜론(;)을 사용한 여러 문장 실행 및 기본 줄바꿈
print("Why,", "Hello"); print("World!")
# Why, Hello
# World!  출력. print하면 \n도 출력


# 구분자(sep)와 끝 문자(end)를 동시에 변경
print("Why,", "Hello", sep="-*-", end="^v^"); print("World!")
# Why,-*-Hello^v^World! 출력

#---------------------------------------------------------------
def square(x):
    return x*x
square(x=10) # 가능

def power(x, exponent=2):
    return x**exponent

power(10) # 가능
power(3,0.5) # 가능
power(2,exponent=3) # 가능
power(x=3,exponent=3) # 가능
power(exponent=3,x=3) # 가능
# power(exponent=3,3) # 불가능
# 오른쪽부터 읽기 때문에...


# ---------------------------------------------------------------------------------
