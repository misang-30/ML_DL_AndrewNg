# ---------------<PL04-Input and Type Conversion>---------------
# >>>1. input()
"""
내장 함수 input()은 문자열(str) 인자를 받을 수 있습니다.
이 문자열은 프롬프트(prompt) 역할을 합니다.
input() 함수가 호출되면, 프로그램은 프롬프트를 화면에 출력하고 
사용자가 키보드로 입력할 때까지 기다립니다.
사용자가 입력을 마치고 Enter(리턴)를 누르면, 
그 입력값이 프로그램으로 전달됩니다.

input()은 항상 사용자의 입력을 문자열(string)로 반환합니다.
숫자 값을 입력받고 싶더라도, 처음에는 문자열로 반환되며, 
숫자로 변환하려면 별도로 처리해야 합니다
"""
name = input("Enter your name : ")


# >>>2. Explicit Type Conversion

str(5) # string 으로 반환
str(1+10) # string 으로 반환
str("hello ") # string을 받을 수도 있다.
str(divmod(14,9)) # tuple도 string으로 바꿀 수 있다.

# >>>3. Evaluating Strings : eval()
"""
eval() 함수는 문자열(str) 인자를 받습니다.
이 문자열을 파이썬 표현식(expression) 으로 평가(evaluate)합니다.
즉, 프로그래머가 직접 코드로 입력한 것처럼 문자열 안의 내용을 실행합니다.
함수는 그 표현식의 결과값을 반환합니다.

"""
str= "3+5"
result = eval(str)
print(str) # "3+5" 출력
print(result) # 8 출력 








