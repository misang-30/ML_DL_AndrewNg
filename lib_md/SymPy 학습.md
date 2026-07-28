

파이썬에서는 **SymPy**라는 라이브러리가 엑셀/파이썬 변수가 아닌 수학적 기호(Symbol)를 다루는 대표적인 수학 라이브러리입니다. 사람이 종이에 미분 문제를 푸는 것처럼 Exact Solution(기호 해)을 구해줍니다.

## 1. SymPy 기본 (수학 기호 미분)

### 1). 변수 선언 및 수식 미분 (`diff`)



```Python
import sympy as sp

# (1). 수학 기호 변수 선언
x, y = sp.symbols('x y')

# (2). 수식 정의: f(x) = x^3 + 2x^2 - 5x + 3
f = x**3 + 2*x**2 - 5*x + 3

# (3). x에 대해 미분 (df/dx)
df = sp.diff(f, x)
print(df)  # 출력: 3*x**2 + 4*x - 5

# (4). 미분한 식에 실제 숫자 대입하기 (x = 2 일 때의 값)
result = df.subs(x, 2)
print(result)  # 출력: 15
```

### 2). 다변수 함수의 편미분

변수가 여러 개인 함수에서 특정 변수 하나로만 미분하는 편미분도 간단하게 처리합니다.



```Python
import sympy as sp

x, y = sp.symbols('x y')

# f(x, y) = x^2 * y^3 + sin(x)
f = x**2 * y**3 + sp.sin(x)

# (1). x에 대해 편미분
df_dx = sp.diff(f, x)
print("x 편미분:", df_dx)  # 2*x*y**3 + cos(x)

# (2). y에 대해 편미분
df_dy = sp.diff(f, y)
print("y 편미분:", df_dy)  # 3*x**2*y**2
```

## 2. 수식을 파이썬/PyTorch 함수로 자동 변환하기 (`lambdify`)

SymPy로 구해낸 기호 미분 결과를 실제 숫자 데이터(NumPy 배열이나 PyTorch 텐서)에 바로 적용할 수 있게 연산용 함수로 바꿀 수 있습니다.


```Python

import sympy as sp
import numpy as np

x = sp.symbols('x')

# (1). SymPy로 복잡한 식 미분하기
f = sp.exp(x) * sp.sin(x)
df = sp.diff(f, x)  # exp(x)*sin(x) + exp(x)*cos(x)

# (2). SymPy 수식을 NumPy용 빠른 파이썬 함수로 변환
f_prime_num = sp.lambdify(x, df, 'numpy')

# (3). 배열 데이터를 넣어서 빠르게 수치 계산
x_vals = np.array([0, np.pi/2, np.pi])
print(f_prime_num(x_vals))  # [1.0, 1.0, -2.3158...]
```

## 3. 딥러닝 자동 미분 vs SymPy 기호 미분 비교

|**구분**|**SymPy (기호 미분, Symbolic)**|**PyTorch / JAX (자동 미분, Autograd)**|
|---|---|---|
|**방식**|사람이 수식 풀듯 수학 공식 자체를 변형|데이터(숫자)가 지나간 연산 그래프 추적|
|**결과물**|미분된 **수학 공식을 문자로 출력** (`3*x**2`)|특정 입력값에서의 **미분 '숫자 값' 출력** (`15.0`)|
|**장점**|오차 없는 정확한 수식 및 이론 검증 가능|행렬/텐서 연산이 매우 빠름, 메모리 효율적|
|**용도**|수학 연구, 논문 수식 검증, 기호 계산|복잡한 인공지능 모델 가중치 학습|

> 딥러닝 연구자들도 논문에 들어갈 복잡한 손실 함수나 수학적 증명을 검증할 때는 **SymPy**로 수식을 먼저 미분해 보고, 구현은 **PyTorch**로 넘기는 방식으로 두 라이브러리를 함께 사용하곤 합니다.