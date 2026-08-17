- 학습 : 훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 획득하는 것을 뜻한다.
- 딥러닝을 end-to-end machine learning 이라고도 부른다. (사람 개입이 없어서)
## 1. 데이터 학습
### 1). 퍼셉트론 수렴 정리
- 선형 분리 가능한 문제라면 데이터로부터 유한 번의 학습을 통해 풀 수 있다.
- 하지만, 비선형 분리 문제는 자동으로 학습할 수 없다.
### 2). 훈련 데이터와 시험 데이터
- 기계학습 문제는 훈련 데이터와 시험 데이터를 나눠 학습과 실험을 수행하는 것이 일반적이다.
- 훈련 데이터 : 학습하면서 최적의 매개변수를 찾습닏.
- 시험 데이터 : 훈련한 모델의 실력을 평가한다. 


---
## 2. 손실 함수
- 신경망 학습에서 사용하는 지표를 손실함수 라고 한다.
- "평균 제곱 오차", "교차 엔트로피 오차"를 사용한다.

### 1).  평균 제곱 오차
``` python
def mean_squared_error(y,t)L
	return 0.5 * np.sum((y-t)**2)
	
# 정답 2 
t = [0,0,1,0,0,0,0,0,0,0]

# 예1 : '2'일 확률이 가장 높다고 추정한다.
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1,0.0, 0.6, 0.0,0.0]
mean_squared_error(np.array(y), np.array(t))



```

### 2). 교차 엔트로피 오차

``` python
def cross_entropy_error(y,t) : 
	delta = 1e-7 # 로그 안이 0이 되지않도록 더해준다.
	return -np.sum(t*np.log(y+delta))

# 정답 2 
t = [0,0,1,0,0,0,0,0,0,0]

# 예1 : '2'일 확률이 가장 높다고 추정한다.
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1,0.0, 0.1, 0.0,0.0]

```
### 3). 미니 배치 학습
- 많은 데이터를 대상으로 일일이 손실 함수를 계산하는 것은 현실적으로 어렵다.
- 이런 경우, 데이터 일부를 추려 전체의 근사치로 이용할 수 있다.
- 신경망 학습에서도 훈련 데이터로부터 일부만 골라 학습을 수행한다.
- 이 일부를 미니배치 라고 한다.
- 가령 60,000장의 훈련 데이터 중에서 100장을 무작위로 뽑아 그 100장 만을 사용하여 학습하는 것이다. 이런 학습 방법을 미니배치 학습이라고 한다.

``` python

# 1. MNIST 데이터셋 읽어오는 코드
import sys, os
sys.path.append(os.pardir)
import numpy as np
from datset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = 
	load_mnist(normalize = True, one_hot_label = True)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

# 2. 훈련 데이터에서 무작위로 10장만 빼내기
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask] # 입력
t_batch = t_train[batch_mask] # 정답

```

``` python
# 배치용 교차 엔트로피 오차 구현하기 

## 1. 원핫 인코딩
def cross_entropy_error(y, t):
    if y.ndim == 1:  # y가 벡터로 주어짐.
        t = t.reshape(1, t.size) # 2차원 데이터 형상으로 바꿔준다.
        y = y.reshape(1, y.size) # 2차원 데이터 형상으로 바꿔준다.
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size
    
## 2. t가 정답 클래스의 인덱스 숫자로 들어올때.
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
    

```

- **`y[np.arange(batch_size), t]` 핵심 작동 원리**:
    
    - `np.arange(batch_size)`는 `[0, 1, 2, ..., batch_size-1]` 배열을 만듭니다.
        
    - numpy 팬시 인덱싱(Fancy Indexing)을 통해 **각 행(데이터)에서 정답 인덱스($t$)에 해당하는 예측 확률만 바로 추출**합니다.
        
    - **예시**: `batch_size = 2`, `t = [2, 0]`인 경우, `y[[0, 1], [2, 0]]`이 실행되어 `y[0, 2]`와 `y[1, 0]`의 값만 배열로 가져옵니다.
        
- **장점**: $0$을 곱하는 무의미한 계산을 생략하고 정답 확률만 바로 뽑아내므로 메모리와 계산 효율이 훨씬 뛰어납니다.


### 4). 왜 손실 함수를 설정하는 가?

> < 지표는 왜 손실함수 쓰는가? >
> 신경망을 학습할 때 정확도를 지표로 삼아서는 안 된다. 정확도를 지표로 하면 매개 변수의 미분이 대부분의 장소에서 0이 되기 때문이다.
> ex) 정확도가 32%인 모델에서 매개변수를 약각만 조정해서는 정확도가 개선되지 않고 일정하게 유지된다. 정확도가 개선되어도 연속적인 값보다는 불연속적인 띄엄띄엄 값으로 바뀌어버린다. 
> 반면 손실함수는 연속적으로 변화한다. 

> < 왜 계단 함수는 활성화 함수로 사용하지 않는가?>
> 계단함수를 활성화 함수로 사용하지 않는 이유도 같다. 
> 계단 함수는 대부분 미분값이 0이다. 이를 사용하면 손실함수를 지표로 삼는 것이 의미가 없다.  매개변수의 작은 변화가 주는 파장을 계단 함수가 말살하여 손실 함수의 값에는 아무런 변화가 나타나지 않기 때문이다.
> 반면, 시그모이드 함수는 함수값도 연속적으로 변화고 곡선의 기울기도 연속적으로 변하며 절대 기울기는 0이 아니다. 따라서 신경망 학습에 사용된다.


---
## 3. 수치 미분
- 경사법에서는 기울기 값을 기준으로 나아갈 방향을 정한다. 

### 1). 미분
``` python
def numerical_diff(f,x): # 수치 미분
	h = 10e-50  # 이것을 쓰면 반올림 오차 때문에 컴퓨터로 계산하는 데 문제가 된다.
	return (f(x+h)-f(x))/h
# 이 미분 방식은 h를 무한히 0으로 좁히는 게 불가능해서 생기는 한계가 있다.
```

``` python
# 대안 : 중심차분(중앙 차분)
def numerical_diff(f,x): # 수치 미분
	h = 10e-50  # 이것을 쓰면 반올림 오차 때문에 컴퓨터로 계산하는 데 문제가 된다.
	return (f(x+h)-f(x-h))/(2*h)

```

> < 해석적 수치 미분 > 
> 수식을 전개해 미분하는 것은 해석적이라는 말을 이용하여 해석적 해 혹은 해석적으로 미분하다 등으로 표현한다. 
> y= x** 2 는 해석적으로 표현하면 dy/dx= 2x로 풀 수 있다.

``` python

def function_1 (x) :
	return 0.01*x ** 2 + 0.1*x

numerical_diff(function_1,5) ## 0.1999 # 실제값 0.2
numerical_diff(function_1, 10) ## 0.29999 # 실제값 0.3

```


### 2). 편미분
- f(x0 , x1) = x_0 ^ 2 + x1 ^ 2
- 변수가 여럿인 함수에 대한 미분을 편미분이라 한다.
``` python 
def function_2(x): # x는 넘파이 배열 가정
	return x[0]**2 + x[1]**2
	# 또는 return np.sum(x**2)
	
```
![](그림4-8.png)

``` python
# x0=3, x1=4 일때 x0에 대한 편미분 구하라.
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

numerical_diff(function_tmp1, 3.0)

# x0=3, x1=4 일때 x1에 대한 편미분 구하라.
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

numerical_diff(function_tmp2, 4.0)

```



---
## 4. 기울기
- 앞서 편미분을 각각의 변수에 대해 했지만, 모든 변수의 편미분을 묶어서 한다고 생각한다.
- 이때 모든 변수의 편미분을 벡터로 정리한 것을 기울기(gradient)라고 한다. 
### 1). 기울기 구하기
``` python
def numerical_gradient(f, x): #f는 함수, x는 넘파이 배열
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size): # 각 원소별로 편미분 진행.
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad
```

``` python
numerical_gradient(function_2, np.array([3.0, 4.0]))
## array( [6.,8.])
numerical_gradient(function_2, np.array([0.0, 2.0]))
## array( [0.,4.])
numerical_gradient(function_2, np.array([3.0, 0.0]))
## array( [6.,0.])
```

### 2). 기울기 그림
- 방향을 가진 벡터로 그려진다.
- 이 그림을 보면 기울기는 함수의 가장 낮은 장소( 최솟값)을 가리키는 것 같다.
- 정확히 말하자면 기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 줄이는 방향
![](그림4-9.png)


### 3). 경사 하강법
- 기울기는 최솟값의 방향을 제시해준다.
- 하지만 최솟값이 있는지 보장 할 수는 없고,실제로 복잡함 함수에서는 기울기가 가리키는 방향에 최솟값이 없는 경우가 대부분이다.

> **WARNING_** 함수가 극솟값, 최솟값, 또 **안장점**(saddle point)이 되는 장소에서는 기울기가 0입니다. 
> 극솟값은 국소적인 최솟값, 즉 한정된 범위에서의 최솟값인 점입니다. 
> 안장점은 어느 방향에서 보면 극댓값이고 다른 방향에서 보면 극솟값이 되는 점입니다.
> 경사법은 기울기가 0인 장소를 찾지만 그것이 반드시 최솟값이라고는 할 수 없습니다(극솟값이나 안장점일 가능성이 있습니다). 
> 또, 복잡하고 찌그러진 모양의 함수라면 (대부분) 평평한 곳으로 파고들면서 **고원**(plateau, 플래토)이라 하는, 학습이 진행되지 않는 정체기에 빠질 수 있습니다.

- 그래서 경사 하강법을 사용한다.
-  경사법은 현 위치에서 기울어진 방향으로 일정 거리만큼 이동합니다. 그런 다음 이동한 곳에서도 마찬가지로 기울기를 구하고, 또 그 기울어진 방향으로 나아가기를 반복합니다. 
- 이렇게 해서 함수의 값을 점차 줄이는 것이 **경사법**(gradient method)입니다. 
- 경사법은 기계학습을 최적화하는 데 흔히 쓰는 방법입니다. 특히 신경망 학습에는 경사법을 많이 사용합니다.


>**NOTE_** 경사법은 최솟값을 찾느냐, 최댓값을 찾느냐에 따라 이름이 다릅니다. 
>전자를 **경사 하강법**(gradient descent method), 후자를 **경사 상승법**(gradient ascent method)

``` python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
    
def function_2(x):
    return x[0]**2 + x[1]**2

# 문제 : 경사 법으로 f(x0,x1) = x0^2 + x1^2 의 최솟값을 구하라.
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
# array([-6.111079e-10, 8.14814391e-10]) 0,0에 가까운 값을 구함.
```

![](그림4-10.png)

- 학습률이 너무 크면 발산하고, 너무 작으면 거의 갱신되지 않은 채 끝나버린다.

### 4). 신경망에서의 기울기
- 아래 식을 가정.
![](그라디언트.png)
- 여기서 common/function.py, common/gradient.py 에서 정의한 함수를 사용한다.

``` python

# coding: utf-8
import os
import sys

# 부모 디렉터리의 모듈을 불러올 수 있도록 파이썬 경로(sys.path)에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.functions import cross_entropy_error, softmax
from common.gradient import numerical_gradient


class SimpleNet:
    """간단한 2입력 3출력 신경망 클래스"""

    def __init__(self):
        # 가중치 매개변수를 정규분포(2x3 형상)로 임의 초기화
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        """입력 데이터 x에 대한 예측값(신경망 출력) 계산"""
        return np.dot(x, self.W)

    def loss(self, x, t):
        """입력 데이터 x와 정답 레이블 t를 받아 손실 함수(Cross Entropy Error) 값 계산"""
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


# --- 실행 코드 ---

# 1. 입력 데이터(x) 및 정답 레이블(t, 원-핫 인코딩 형태) 정의
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

# 2. 신경망 객체 생성
net = SimpleNet()

# 3. 수치 미분을 위한 람다(Lambda) 함수 정의
#    numerical_gradient 함수 내부에서 f(w) 형태의 단일 인자 함수를 호출하므로 더미 변수 w를 받도록 설정
f = lambda w: net.loss(x, t)

# 4. 가중치 매개변수 W에 대한 기울기(Gradient) 계산
dW = numerical_gradient(f, net.W)

# 5. 결과 출력
print(dW)
```



---
## 5. 학습 알고리즘 구현하기

### 1). 신경망 학습의 절차
```
**전제** 신경망에는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 '학습'이라 합니다. 신경망 학습은 다음과 같이 4단계로 수행합니다.

**1단계 - 미니배치** 훈련 데이터 중 일부를 무작위로 가져옵니다. 이렇게 선별한 데이터를 미니배치라 하며, 그 미니배치의 손실 함수 값을 줄이는 것이 목표입니다. (확률적 경사하강법)

**2단계 - 기울기 산출** 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구합니다. 기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시합니다.

**3단계 - 매개변수 갱신** 가중치 매개변수를 기울기 방향으로 아주 조금 갱신합니다.

**4단계 - 반복** 1~3단계를 반복합니다.
```

### 2). 2층 신경망 클래스 구현

``` python
# coding: utf-8
import os
import sys

# 부모 디렉터리의 모듈을 불러올 수 있도록 sys.path에 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.functions import cross_entropy_error, sigmoid, sigmoid_grad, softmax
from common.gradient import numerical_gradient


class TwoLayerNet:
    """2층 신경망(1개의 은닉층, 1개의 출력층) 클래스"""

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """가중치 및 편향 매개변수 초기화

        - input_size: 입력층의 뉴런 수 (예: MNIST 이미지 784)
        - hidden_size: 은닉층의 뉴런 수
        - output_size: 출력층의 뉴런 수 (예: 숫자 클래스 10)
        - weight_init_std: 가중치 초기화 시 사용할 정규분포의 표준편차
        """
        self.params = {}
        # 1층(입력층 -> 은닉층) 가중치 및 편향
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)

        # 2층(은닉층 -> 출력층) 가중치 및 편향
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        """입력 데이터 x에 대한 순전파(예측값 산출) 수행"""
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        # 1층 연산: 행렬 곱 + 편향 합 -> 시그모이드 활성화 함수
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        # 2층 연산: 행렬 곱 + 편향 합 -> 소프트맥스 출력 함수
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """손실 함수(교차 엔트로피 오차) 값 계산

        - x: 입력 데이터, t: 정답 레이블
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """모델의 분류 정확도 측정

        - x: 입력 데이터, t: 정답 레이블
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        # 정답 레이블 t가 원-핫 인코딩 형태일 경우 차원 축소
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """수치 미분을 사용한 각 매개변수별 기울기 계산 (검증용, 속도가 느림)"""
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

    def gradient(self, x, t):
        """오차역전파법(Backpropagation)을 사용한 고속 기울기 계산"""
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        grads = {}

        batch_num = x.shape[0]

        # 1. 순전파 (Forward)
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # 2. 역전파 (Backward)
        # 2층(소프트맥스 + 교차 엔트로피 오차) 오차 전파
        dy = (y - t) / batch_num
        grads["W2"] = np.dot(z1.T, dy)
        grads["b2"] = np.sum(dy, axis=0)

        # 1층(시그모이드) 오차 전파
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads["W1"] = np.dot(x.T, dz1)
        grads["b1"] = np.sum(dz1, axis=0)

        return grads
```



### 3). 미니 배치 학습 구현하기
``` python
# coding: utf-8
import os
import sys

# 부모 디렉터리의 모듈을 불러올 수 있도록 sys.path에 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 1. MNIST 데이터셋 로드
# normalize=True: 이미지 피셀 값을 0.0~1.0 사이로 정규화
# one_hot_label=True: 정답 레이블을 원-핫 인코딩 배열 형태로 반환
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True
)

# 학습 경과(손실 값) 기록용 리스트
train_loss_list = []

# 2. 하이퍼파라미터 설정
iters_num = 10000  # 경사 하강법 반복 횟수
train_size = x_train.shape[0]  # 훈련 데이터 총 개수 (60,000개)
batch_size = 100  # 미니배치 크기
learning_rate = 0.1  # 학습률 (Learning Rate)

# 3. 2층 신경망 모델 객체 생성 (입력: 784, 은닉: 50, 출력: 10)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 4. 학습 루프 실행
for i in range(iters_num):
    # ① 미니배치 획득: 전체 훈련 데이터 중 batch_size만큼 무작위 추출
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # ② 기울기 계산 (오차역전파법으로 고속 계산)
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식 (느림)
    grad = network.gradient(x_batch, t_batch)

    # ③ 매개변수 갱신 (경사 하강법: W = W - lr * dW)
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # ④ 학습 경과 기록 (손실 함수 값 저장)
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

# 5. 학습 진행 과정(Loss 변화) 시각화
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.ylim(0, 9)
plt.xlim(0, 10000)
plt.show()
```
![](그림4-11.png)

### 4). 시험 데이터로 평가하기

> [NOTE] 에폭(epoch)은 하나의 단위입니다. 1에폭은 학습에서 훈련 데이터를 모두 소진했을 때의 횟수에 해당한다.
> ex : 10,000 개를 100개의 미니배치로 학습할 경우, SGD는 100회 반복하면 모든 훈련 데이터를 소진한다.  이때, 1 epoch은 100회이다.
> 

``` python
# coding: utf-8
import os
import sys

# 부모 디렉터리의 모듈을 불러올 수 있도록 sys.path에 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 1. MNIST 데이터셋 로드
# normalize=True: 이미지 피셀 값을 0.0~1.0 사이로 정규화
# one_hot_label=True: 정답 레이블을 원-핫 인코딩 배열 형태로 반환
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True
)

# 2. 2층 신경망 모델 객체 생성 (입력: 784, 은닉: 50, 출력: 10)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 3. 하이퍼파라미터 설정
iters_num = 10000  # 총 반복 횟수
train_size = x_train.shape[0]  # 훈련 데이터 총 개수 (60,000개)
batch_size = 100  # 미니배치 크기
learning_rate = 0.1  # 학습률 (Learning Rate)

# 학습 경과(Loss 및 정확도) 기록용 리스트
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1 에폭(Epoch)당 반복 수 계산 (60000 / 100 = 600회)
iter_per_epoch = max(train_size / batch_size, 1)

# 4. 학습 루프 실행
for i in range(iters_num):
    # ① 미니배치 획득: 전체 훈련 데이터 중 batch_size만큼 무작위 추출
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # ② 기울기 계산 (오차역전파법으로 고속 계산)
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식 (느림)
    grad = network.gradient(x_batch, t_batch)

    # ③ 매개변수 갱신 (경사 하강법)
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # ④ 손실 함수 값 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # ⑤ 1 에폭마다 훈련 데이터와 시험 데이터의 정확도 평가 및 기록
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {train_acc:.4f}, {test_acc:.4f}")

# 5. 에폭별 정확도(Accuracy) 시각화
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label="train acc")
plt.plot(x, test_acc_list, label="test acc", linestyle="--")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.xlim(0, 16)
plt.legend(loc="lower right")
plt.show()
```
![](그림4-12.png)

