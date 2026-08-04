- Week 2: Regression with multiple input variables
# < Multiple Regression>

## 1. Multiple Features
- Multiple Linear Regression

- f w,b(x) = w1x1 + ... + w_n * x_n + b
-  f w,b(x) = w벡터 * x벡터 + b 

```math
\vec{w} = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix}, \quad \vec{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
```

```math
\vec{w}^T = \begin{bmatrix} w_1 & w_2 & \dots & w_n \end{bmatrix}
```


```math
f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b
```

---
---


## 2. Vectorization 
- 코드를 줄여주고 효율성을 극대화할 수 있다.
- np.dot 함수는 동시에 계산하는 것이다.
- 동시에 계산해서 더하는 것이라서 동작 시간이 짧다.
- ==Numpy 라이브러리== 학습 하라.
``` python
## 벡터화 없는 버전

# way 1
f = w[0] * x[0] +
	w[1] * x[1] +
	w[2] * x[2] + b
	
# way 2 
f = 0
for j in range(0,n):
	f = f + w[j] * x[j]
f = f + b

```

``` python
## 벡터화 버전
f = np.dot(w,x) + b 
```

### 1). 예시
``` python

w = np.array([0.5, 1.3,..., 3.4])
d = np.array([0.3, 0.2 , ... , 0.4])

## 벡터화 없는 버전
for j in range(0,16):
	w[j] = w[j] - 0.1 * d[j]

## 벡터화 있는 버전
w = w - 0.1 * d

```

---
---
## 3. GD for Multiple Regression

### 1). Feature 1개
repeat {
  $$w = w - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}$$
  
  $$\text{where } \frac{\partial}{\partial w} J(w,b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}$$

  $$b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)$$

  simultaneously update $w, b$
}

### 2). $n$ features ($n \ge 2$)
repeat {
  $$w_1 = w_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_1^{(i)}$$
  $$\vdots$$
  $$w_n = w_n - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_n^{(i)}$$

  $$b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)$$

  simultaneously update $w_j \text{ (for } j=1, \dots, n\text{)} \text{ and } b$
}

---
### 3). Normal Equation
- 대안으로 Normal Equation(정규 방정식)이 있다. 
- 이것에 대해 깊은 설명은 하지 않는다.
- 장점
	반복없이 w,b를 구할 수 있다.
	Linear Regression에서만 사용 가능하다.
- 단점
	- 다른 학습 알고리즘에 일반화해서 적용이 안된다.
	- Feature가 많으면 느리다.
	- Normal Equation은 머신러닝 라이브러리 내에서 쓰일  수 있다.

---
---
---
# < Practical Tips ofr Linear Regression>

## 1. Feature Scaling
- Feature Size가 너무 작거나 크면 w의 크기가 너무 작거나 커지고,
- ScatterPlot과 Contour Plot에서 각 데이터를 구분하기 힘들다.
- 또한, 서로 다른 특성들이 너무 다른 범위의 값을 가질 때 경사 하강법은 속도가 느려질 수 있습니다. 
- 하지만 특성들을 재조정하여 모두 비슷한 범위를 갖도록 스케일링해주면 경사 하강법의 속도를 획기적으로 향상시킬 수 있습니다.
- 그래서 Feature Scaling을 해야 한다. 
### 1).Feature Scaling 방법
- 1. Scaling By Maximum : Feature의 최대값으로 나누어서 1 아래 값으로 스케일 하는 방법이 있다.
- 2.Mean Normalization : 평균값으로 나누어서 Feature가 음수와 양수 범위에서 동작하게 할 수 있다.
- 3.Z-Score Normalization : 표준편차 이용해서 하는 방법

### 2).  Rule of Thumb ( 스케일링 적용 기준)
|**상태**|**특성 값 범위 (예시)**|**스케일링 필요 여부**|
|---|---|---|
|**적절함**|$-3 \sim +3$, $-0.3 \sim +0.3$, $0 \sim 3$, $-2 \sim +0.5$|**불필요** (스케일링을 안 해도 잘 작동함)|
|**너무 큼**|$-100 \sim +100$, $98.6 \sim 105$ (체온 측정값 등)|**필수/권장** (경사 하강법 속도를 저하시킴)|
|**너무 작음**|$-0.001 \sim +0.001$|**필수/권장**|

---
---
## 2. Checking GD for Convergence
- GD가 수렴하는지 체크하는 방법
### 1). 학습 곡선 활용하는 방법
- 가장 직관적이고 추천하는 방법은 학습 곡선(Learning Curve)을 그려보는 것입니다.
- **그래프 축 설정**:
    - **가로축(X축)**: 경사 하강법 **반복 횟수 (Number of Iterations)**
    - **세로축(Y축)**: 비용 함수 **$J(w, b)$ 값**
        
- **정상 작동 시 모양**:
    
    - 경사 하강법이 제대로 작동한다면, **매 반복(Iteration)마다 비용 $J$가 지속적으로 감소**해야 합니다.
        
    - 반복 횟수가 쌓임에 따라 곡선이 점차 평평해지며(Leveling off/Flattening), 더 이상 $J$가 감소하지 않는 지점에 도달하면 **수렴(Converged)한 것으로 판단**합니다.
        
- **이상 징후**:
    - 만약 중간에 비용 $J$가 증가하는 구간이 있다면, **학습률 $\alpha$가 너무 크게 설정**되었거나 **코드에 버그**가 있을 가능성이 높습니다.
        
> **참고**: 경사 하강법이 수렴하는 데 필요한 반복 횟수(예: 30회, 1,000회, 100,000회 등)는 데이터셋과 문제에 따라 크게 다르므로, 미리 예측하기 어려워 이 곡선을 확인하는 것이 필수적입니다.

### 2). 자동 수렴 검사
- 그래프를 직접 보지 않고 코드상에서 수렴 여부를 자동으로 판단하는 방법입니다.

- **임계값 $\epsilon$ (Epsilon) 설정**: 아주 작은 값(예: $10^{-3} = 0.001$)을 지정합니다.
    
- **판단 기준**: 한 번의 반복(Iteration) 동안 **비용 $J$의 감소량이 $\epsilon$보다 적으면** 수렴한 것으로 간주하고 학습을 종료합니다.

---
---
## 3. Choosing the Learning Rate 

### 1). Learning Rate가 적절하지 않을 때, 현상
- **학습률이 너무 큰 경우 ($\alpha$ is too large)**:
    - **증상 1**: 비용 함수 $J$가 감소하다가 늘어나는 등 오르락내리락(Bouncing)합니다.
    - **증상 2**: 매 반복(Iteration)마다 비용 $J$가 **지속적으로 증가**합니다.
    - **원인**: 최솟값을 지나쳐 튀어나가는 **오버슈팅(Overshooting)** 현상이 발생하기 때문입니다.
        
- **학습률이 너무 작은 경우 ($\alpha$ is too small)**:
    - **증상**: 비용 $J$는 지속적으로 잘 감소하지만, **수렴하는 데 너무 많은 반복 횟수가 걸려 매우 느립니다.**

### 2). 코드 디버깅 팁
- **원칙**: 학습률 $\alpha$가 **충분히 작다면, 모든 반복(Iteration)마다 비용 $J$는 무조건 감소**해야 합니다.
    
- **디버깅 과정**:
    1. 경사 하강법이 이상하게 작동한다면, **$\alpha$를 매우매우 작은 값으로 설정**해 봅니다.
        
    2. 만약 아주 작은 $\alpha$에서도 비용 $J$가 매번 감소하지 않고 증가한다면, 이는 학습률 문제가 아니라 코드 상의 버그(Bug)입니다.
        
    3. **자주 발생하는 버그 예시**: 가중치 업데이트 공식에서 뺄셈(`-`) 대신 실수로 더하기(`+`)를 작성한 경우 ($w_1 = w_1 + \alpha \times \text{derivative}$).
        
        - 더하기를 하면 최솟값에서 점점 멀어지므로 비용이 지속적으로 증가합니다.
            
- **주의사항**: 아주 작은 $\alpha$ 설정은 오직 **코드 디버깅 목적**으로만 사용해야 하며, 실제 모델 학습 시에는 비효율적입니다.

### 3). 최적의 Learning Rate 찾는 팁 (3배수 법칙)

- **약 3배씩 차이나는 범위의 후보군 설정**:
    
    - 예: `0.001` $\rightarrow$ `0.003` $\rightarrow$ `0.01` $\rightarrow$ `0.03` $\rightarrow$ `0.1` $\rightarrow$ `0.3` $\rightarrow$ `1` ...
        
- **짧은 반복 테스트**:
    
    - 각 $\alpha$ 값으로 경사 하강법을 **적은 횟수의 Iteration만 실행**해 봅니다.
        
- **학습 곡선 시각화**:
    
    - 각 $\alpha$에 따른 `반복 횟수 대비 비용 $J$` 그래프를 그려봅니다.
        
- **최종 선택**:
    
    - 비용 $J$가 가장 빠르면서도 안정적으로 지속 감소하는 $\alpha$를 선택합니다.
        
    - 실전에서는 **비용이 발산하지 않는 최대 $\alpha$ 값보다 살짝 작은 값**을 최종 학습률로 지정합니다. (학습을 가장 빠르게 만들 수 있는 최대한 큰 보폭을 찾고 그것보다 조금 작게 하는 방식.)

---
---
## 4.Feature Engineering
- **개념**: 기존에 주어진 데이터(특성, Features)를 사용자의 직관이나 전문 지식을 바탕으로 변합·조합하여 **새로운 특성**을 만들어내는 기법입니다.
### 1). Polynomial Regression (다항 회귀)
단순한 직선($f(x) = wx + b$)으로는 표현하기 어려운 **비선형(곡선) 데이터 패턴**을 맞추기 위해, 기존 특성(Feature)에 **2차, 3차 등 제곱 항을 추가**하여 곡선 형태로 모델링하는 방법입니다.

- **2차 함수 (Quadratic):** $f(x) = w_1 x + w_2 x^2 + b$
    
    - _문제점:_ 곡선이 올라갔다가 결국 다시 내려옵니다. (집 평수가 넓어지는데 집값이 떨어지는 기현상 발생)
        
- **3차 함수 (Cubic):** $f(x) = w_1 x + w_2 x^2 + w_3 x^3 + b$
    
    - _장점:_ 집값이 상승하다가 일정 시점 이후 계속 상승하는 실제 패턴을 더 잘 반영합니다.
        
- **제곱근 함수 (Square Root):** $f(x) = w_1 x + w_2 \sqrt{x} + b$
    
    - _장점:_ 집값이 계속 올라가지만 상승 폭이 점점 완만해지는 특성을 표현하기에 아주 좋습니다.
### 2).다항 회귀 사용 시 필수 주의사항
- 특성 스케일링 (Feature Scaling)

- 제곱/세제곱 항을 추가하면 **특성 간의 값 범위(Scale) 격차가 극심**해집니다.

> **예시 (집 크기 $x = 1 \sim 1,000$ $\text{ft}^2$ 기준)**
> 
> - $x$ (1차): $1 \sim 1,000$
>     
> - $x^2$ (2차): $1 \sim 1,000,000$ (1백만)
>     
> - $x^3$ (3차): $1 \sim 1,000,000,000$ (10억)
>     

- 이처럼 범위 차이가 크면 경사 하강법(Gradient Descent)이 발산하거나 수렴하는 데 극도로 오랜 시간이 걸리므로, **반드시 Feature Scaling(정규화/표준화)을 적용해야 합니다.**

### 3). Scikit-Learn
- 전 세계 머신러닝 실무에서 가장 널리 쓰이는 파이썬 라이브러리입니다.
    
- 복잡한 알고리즘을 단 몇 줄의 코드로 구현할 수 있습니다.
    
- _교수의 조언:_ 블랙박스처럼 라이브러리만 쓰기보다, internal 동작 원리를 직접 구현해보며 이해하는 것이 매우 중요합니다.
