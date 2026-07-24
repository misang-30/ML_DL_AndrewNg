- Classification 다룬다.

## < Motivation>
### 1. 분류 문제(Classification)와 이진 분류(Binary Classification)

- **분류(Classification):** 예측하려는 결과값($y$)이 연속된 숫자가 아니라 **몇 가지 정해진 카테고리(범주)** 중 하나인 문제입니다.
    
- **이진 분류(Binary Classification):** 결과 카테고리가 단 2개(예: Yes/No, True/False)인 분류 문제입니다.
    
    - **음성 클래스 (Negative Class / 0):** 대상의 부재 (예: 스팸 아님, 정상 거래, 양성 종양)
        
    - **양성 클래스 (Positive Class / 1):** 대상의 존재 (예: 스팸, 사기 거래, 악성 종양)
        
    - _참고:_ 'Positive/Negative'라는 명칭은 긍정/부정을 뜻하는 것이 아니라, 찾고자 하는 특성이 존재하는지 여부를 나타냅니다.
        
---


### 2. 왜 선형 회귀(Linear Regression)는 분류에 적합하지 않을까?

스팸 감지나 악성 종양 판단에 선형 회귀를 그대로 적용하고, 임계값(Threshold, 예: 0.5)을 기준으로 분류하려 하면 문제가 발생합니다.

####  이상치(Outlier)에 지나치게 민감함

1. 종양 크기가 아주 큰 데이터(이상치) 하나가 우측 멀리에 추가되었다고 가정합니다.
    
2. 기존 데이터의 분류 기준은 그대로여야 정상이지만, 선형 회귀의 직선은 이 하나의 데이터 때문에 **기울기와 위치가 크게 이동**해 버립니다.
    
3. 그 결과 **분류 경계선(Decision Boundary)이 밀려나면서**, 기존에 잘 맞추던 데이터들까지 오분류(False classification)하게 됩니다.
    

> **요약:** 선형 회귀는 출력값이 $-\infty$부터 $+\infty$까지 나올 수 있어서, $0$과 $1$ 사이의 카테고리를 예측하는 분류 문제에는 치명적인 오차가 생깁니다.

---
### 3. 해결책: 로지스틱 회귀 (Logistic Regression)

- **주의할 점:** 이름에 'Regression(회귀)'이 들어가 있지만, 실제로는 **분류(Classification) 문제**를 해결하는 알고리즘입니다. (역사적인 이유로 이름이 이렇게 붙었습니다.)
    
- **특징:** 출력값이 항상 **$0$과 $1$ 사이**로 제한되므로, 이상치가 추가되어도 분류 경계선이 크게 흔들리지 않고 안정적으로 예측할 수 있습니다.
---
---
---
## < Logistic Regression >
### 1. 로지스틱 회귀(Logistic Regression)란?

- 전 세계에서 **가장 널리 쓰이는 대표적인 분류(Classification) 알고리즘**입니다.
    
- 선형 회귀의 직선 대신 S자 형태의 곡선(S-shaped curve)을 데이터에 맞춰 $0$과 $1$ 사이의 값으로 예측을 수행합니다.
    
---

### 2. 핵심 수학적 도구: 시그모이드 함수 (Sigmoid Function)

선형 회귀의 출력값을 $0$과 $1$ 사이의 범위로 압축해주기 위해 **시그모이드 함수**(로지스틱 함수라고도 함)를 사용합니다.

#### 수식
$$g(z) = \frac{1}{1 + e^{-z}}$$

- $e$: 자연상수 ($\approx 2.718$)
    
- $z$: 선형 회귀의 수식인 $z = \mathbf{w} \cdot \mathbf{x} + b$
    

#### 특징 및 동작 원리

- **$z$가 매우 클 때 ($z \to \infty$):** $e^{-z} \to 0$이 되므로 $g(z) \approx \frac{1}{1 + 0} = \mathbf{1}$
    
- **$z$가 매우 작을 때 ($z \to -\infty$):** $e^{-z}$가 매우 커져 분모가 거대해지므로 $g(z) \approx \mathbf{0}$
    
- **$z = 0$일 때:** $e^{0} = 1$이 되므로 $g(z) = \frac{1}{1 + 1} = \mathbf{0.5}$ (y축 절편)
    
---
### 3. 로지스틱 회귀의 전체 모델 방정식

- 선형 회귀식 $z$를 시그모이드 함수 $g(z)$의 입력으로 넣어 합친 형태입니다.

 - $$f_{\mathbf{w},b}(\mathbf{x}) = g(\mathbf{w} \cdot \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$$
---

### 4. 모델 출력값의 해석: 조건부 확률 $P(y=1\vert{}\mathbf{x})$

로지스틱 회귀의 출력값은 단순한 $0$이나 $1$이 아니라, "특정 입력 $\mathbf{x}$가 주어졌을 때 결과 $y$가 1(Positive Class)일 확률"로 해석합니다.

> **예시 (종양 크기 기반 악성 여부 판단):**
> 
> - 입력 $\mathbf{x}$(종양 크기)에 대해 모델이 **$0.7$**을 출력했다면?
>     
>     - 해당 종양이 악성($y=1$)일 확률이 **70%**라는 의미입니다.
>         
>     - $y$는 $0$ 아니면 $1$이므로, 자동으로 정상($y=0$)일 확률은 **$1 - 0.7 = 0.3$ (30%)**가 됩니다.


---
---
---
## < Decision Boundary >
### 1. 예측 분류 기준과 임계값(Threshold)

로지스틱 회귀 모델의 출력값 $f(\mathbf{x}) = g(z)$는 **클래스 1이 될 확률**을 나타냅니다. (단, $z = \mathbf{w} \cdot \mathbf{x} + b$)

실제 예측값 $\hat{y}$를 $0$ 또는 $1$로 결정할 때는 보통 $0.5$를 기준(Threshold)으로 잡습니다.

- **$f(\mathbf{x}) \ge 0.5$** 이면 $\Rightarrow$ **$\hat{y} = 1$** 로 예측
- **$f(\mathbf{x}) < 0.5$** 이면 $\Rightarrow$ **$\hat{y} = 0$** 으로 예측
    
---
### 2. 시그모이드 함수 조건과 $z$의 관계

시그모이드 함수 $g(z)$의 특성에 따라, $g(z) \ge 0.5$가 되는 조건은 **$z \ge 0$** 일 때입니다.
즉, 분류 예측 기준을 $z$ 수식으로 직접 풀어 쓰면 다음과 같습니다.

```math
\mathbf{w} \cdot \mathbf{x} + b \ge 0 \implies \hat{y} = 1
```

```math
\mathbf{w} \cdot \mathbf{x} + b < 0 \implies \hat{y} = 0
```

> **결론:** $y=1$과 $y=0$을 가르는 경계선은 바로 **$\mathbf{w} \cdot \mathbf{x} + b = 0$** 이 되는 지
> 점입니다. 이를 **결정 경계(Decision Boundary)**라고 부릅니다.
- z = 0 = w x +b 일때, 결정 경계를 구할 수 있다.

---
### 3. 결정 경계(Decision Boundary)의 형태

#### 1) 선형 결정 경계 (Linear Decision Boundary)

고차 항 없이 기본적인 입력 특성($x_1, x_2, \dots$)만 사용하면 결정 경계는 항상 **직선(또는 평면)** 형태가 됩니다.

> **예시 수식:** $x_1 + x_2 - 3 = 0 \implies x_1 + x_2 = 3$
> 
> - 이 직선의 우상단 영역은 $\hat{y} = 1$, 좌하단 영역은 $\hat{y} = 0$으로 분류됩니다.
>     

#### 2) 비선형 결정 경계 (Non-linear Decision Boundary)

다항 회귀(Polynomial Regression)처럼 제곱 항이나 교차 항($x_1^2, x_2^2, x_1 x_2$ 등)을 특성 공학(Feature Engineering)으로 추가하면 복잡한 곡선 형태의 경계선도 만들어낼 수 있습니다.

- **원형 경계:** $x_1^2 + x_2^2 - 1 = 0 \implies x_1^2 + x_2^2 = 1$
    
    - 원의 **외부**는 $\hat{y} = 1$, **내부**는 $\hat{y} = 0$으로 분류됩니다.
        
- **고차 다항식:** 타원, 복잡한 곡선, 혹은 닫힌 곡면 형태 등 데이터 분포에 맞춰 매우 정교하고 자유로운 결정 경계를 학습할 수 있습니다.

---
---
---
## < Cost Function >

### 1. Cost Function for Logistic Regression

#### 0). Convex Vs Non-Convex
- Convex : 볼록하다는 뜻이다.
- Convex Function은 볼록 함수를 의미한다.
- **Convex (볼록 함수 / 아래로 볼록):**
    - 모양이 매끄러운 밥그릇(U자)이나 **미끄럼틀** 형태입니다.
    - 어디서 공을 굴려도 결국 하나의 가장 낮은 지점(최저점)으로만 떨어집니다.
- **Non-Convex (비볼록 함수):**
    - 오르락내리락하는 **롤러코스터**나 산맥 형태입니다.
    - 진짜 가장 낮은 골짜기(Global Minimum)가 아닌, 어중간한 언덕 사이 골짜기(Local Minimum)에 공이 갇힐 위험이 있습니다.

---
#### 1). 기존 제곱 오차(Squared Error) 손실 함수의 문제점

선형 회귀에서 사용하던 제곱 오차 손실 함수($\frac{1}{2}(f(\mathbf{x}) - y)^2$)를 로지스틱 회귀($f(\mathbf{x}) = \frac{1}{1 + e^{-z}}$)에 그대로 적용하면 손실 함수 그래프가 Non-Convex(울퉁불퉁한 비볼록 형태)가 됩니다.

- **문제 발생:** 언덕과 골짜기(Local Minima)가 너무 많아져서, 경사 하강법(Gradient Descent)을 실행할 때 **진짜 최저점(Global Minimum)에 도달하지 못하고 중간 골짜기에 갇히는 현상**이 발생합니다.
    
---
#### 2). 해결책: 로지스틱 회귀 전용 손실 함수 (Loss Function)

경사 하강법을 적용했을 때 무조건 최저점으로 수렴하는 **Convex(밥그릇 형태)** 함수로 만들기 위해 로그 함수를 활용한 새로운 손실 함수 $L(f(\mathbf{x}), y)$를 정의합니다.

> **단일 학습 데이터에 대한 손실 함수 정의:**

```math
L(f(\mathbf{x}), y) = \begin{cases} 
-\log(f(\mathbf{x})) & \text{if } y = 1 \\
-\log(1 - f(\mathbf{x})) & \text{if } y = 0 
\end{cases}
```


![[Pasted image 20260721132927.png]]

##### 실제 정답 $y = 1$ 인 경우

- 모델 예측값 $f(\mathbf{x})$가 **$1$에 가까워지면** $\rightarrow$ **손실(Loss)은 $0$에 가까워짐** (잘 맞췄으므로 벌점이 없음).
    
- 모델 예측값 $f(\mathbf{x})$가 $0$에 가까워지면 $\rightarrow$ 손실(Loss)은 무한대($\infty$)로 치솟음 (정답이 1인데 0에 가깝다고 예측했으므로 매우 강력한 페널티를 부여).

##### 실제 정답 $y = 0$ 인 경우

- 모델 예측값 $f(\mathbf{x})$가 **$0$에 가까워지면** $\rightarrow$ **손실(Loss)은 $0$에 가까워짐** (잘 맞췄으므로 벌점이 없음).
    
- 모델 예측값 $f(\mathbf{x})$가 $1$에 가까워지면 $\rightarrow$ 손실(Loss)은 무한대($\infty$)로 치솟음 (정답이 0인데 1에 가깝다고 예측했으므로 매우 강력한 페널티를 부여).
    
---
#### 3). 전체 데이터셋에 대한 비용 함수 (Cost Function)

전체 $m$개 데이터셋에 대한 총 비용 함수 $J(\mathbf{w}, b)$는 각 개별 데이터 손실값의 **평균**입니다.

```math
J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^{m} L(f(\mathbf{x}^{(i)}), y^{(i)})
```

- 이 손실 함수를 합쳐서 만든 비용 함수 $J(\mathbf{w}, b)$는 **매끄러운 Convex 형태**를 띠게 되므로, 경사 하강법을 사용하면 **항상 전역 최저점(Global Minimum)을 안전하게 찾을 수 있음이 보장**됩니다.

![[Pasted image 20260721131134.png]]

---
---
### 2. Simplified Cost Function

#### 1). 손실 함수, 비용 함수 정리
- 앞서 다룬 손실 함수를 하나로 합친다.
$$L(f(\mathbf{x}), y) = -y \log(f(\mathbf{x})) - (1 - y) \log(1 - f(\mathbf{x}))$$

- 따라서 Cost Function은 아래와 같다.
- 이 비용 함수는 Convex(볼록한 밥그릇 모양)이므로, 경사 하강법(Gradient Descent)을 사용하면 최적의 정답($\mathbf{w}, b$)에 안전하게 도달합니다.
```math
J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\left(f_{\mathbf{w},b}(\mathbf{x}^{(i)})\right) + (1 - y^{(i)}) \log\left(1 - f_{\mathbf{w},b}(\mathbf{x}^{(i)})\right) \right]
```
#### 2). 이 손실 함수를 사용하는 이유: MLE (최대 우도 추정)

이 복잡해 보이는 로그 손실 함수는 임의로 만들어낸 것이 아닙니다.

- 통계학의 Maximum Likelihood Estimation (MLE, 최대 우도 추정)이라는 원리에서 수학적으로 유도된 공식입니다.
    
- 우리가 수집한 실제 데이터가 발생할 확률을 **가장 크게 만들어주는 파라미터($\mathbf{w}, b$)를 찾는 정교한 수학적 근거**를 가지고 있습니다.

---
---
---
## < Gradient Descent for Logistic Regression>
### 1. 로지스틱 회귀의 경사 하강법 (Gradient Descent)

비용 함수 $J(\mathbf{w}, b)$를 최소화하는 파라미터 $\mathbf{w}$와 $b$를 찾기 위해, 매 단계마다 파라미터를 동시에(Simultaneously) 업데이트합니다.

#### 파라미터 업데이트 수식

- **가중치($w_j$) 업데이트:**
```math
w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}
```
    
- **편향($b$) 업데이트:**
```math
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \right)
```

_(단, $\alpha$는 학습률(Learning Rate), $m$은 전체 데이터 개수)_

### 2.  선형 회귀 vs 로지스틱 회귀: 착각하기 쉬운 핵심 포인트

위 업데이트 공식을 보면 **선형 회귀(Linear Regression)의 경사 하강법 공식과 형태가 100% 완벽히 동일**해 보입니다.

하지만 **두 알고리즘은 엄연히 완전히 다른 알고리즘**입니다. 그 이유는 예측 함수 **$f(\mathbf{x})$의 정의**가 다르기 때문입니다.

|**구분**|**선형 회귀 (Linear Regression)**|**로지스틱 회귀 (Logistic Regression)**|
|---|---|---|
|**$f(\mathbf{x})$ 정의**|$f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$ (직선)|$f(\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$ (**시그모이드 적용**)|
|**출력 범위**|$-\infty \sim +\infty$ (연속적인 값)|$0 \sim 1$ (확률값)|

### 3. 학습 속도를 높이는 두 가지 핵심 팁

1. **특성 스케일링 (Feature Scaling):**
    
    - 입력 특성들의 범위를 비슷하게($-1 \sim +1$ 등) 맞춰주면, 로지스틱 회귀에서도 경사 하강법이 수렴하는 속도가 훨씬 빨라집니다.
        
2. **벡터화 (Vectorization):**
    
    - 반복문(`for` loop) 대신 행렬 연산을 활용(Vectorization)하면 계산 속도를 극적으로 향상시킬 수 있습니다.

---
---
---

## < Regularization to Reduce Overfitting>

### 1. The Problem of Overfitting
- Underfit : High Bias
- Generalization : 일반화
- Overfit : high Varirance (학습 데이터에 너무 학습된 경우)
#### Overfitting vs Underfitting

머신러닝 모델을 학습시킬 때 발생할 수 있는 주요 문제인 과소적합(Underfitting)과 **과대적합(Overfitting)**, 그리고 우리가 지향해야 하는 적절한 모델(Just Right)의 차이점입니다.

| **구분**        | **과소적합 (Underfitting)**        | **적절한 모델 (Just Right)**      | **과대적합 (Overfitting)**          |
| ------------- | ------------------------------ | ---------------------------- | ------------------------------- |
| **다른 용어**     | **High Bias (높은 편향)**          | Good Generalization (좋은 일반화) | **High Variance (높은 분산)**       |
| **특징**        | 데이터의 패턴을 거의 학습하지 못함            | 데이터의 전체적인 경향성을 잘 포착함         | 훈련 데이터의 아주 미세한 오차/노이즈까지 전부 외워버림 |
| **훈련 데이터 성능** | 나쁨 (오차가 큼)                     | 적절함                          | 매우 좋음 (비용 함수 $J \approx 0$)     |
| **새 데이터 예측**  | 나쁨                             | **우수함 (일반화 성능 높음)**          | 나쁨 (일반화 성능 떨어짐)                 |
| **원인**        | 특성(Feature)이 너무 적거나 모델이 너무 단순함 | 적절한 수의 특성과 복잡도               | 특성이 너무 많거나(고차 다항식) 모델이 너무 복잡함   |

#### Underfitting (과소적합) / High Bias (높은 편향)

- **개념:** 모델이 훈련 데이터의 기본적인 패턴조차 제대로 반영하지 못하는 상태입니다.
    
- **이유:** 모델에 대한 강한 사전 편견(Preconception/Bias)을 가지고 있기 때문입니다.
    
    - _예시:_ 집값 데이터가 곡선 형태를 띠고 있음에도 "집값은 무조건 직선(선형 함수) 형태일 것"이라는 강한 편견으로 선형 모델을 적용할 때 발생합니다.
        
- **결과:** 훈련 데이터에서도, 새로운 데이터에서도 모두 성능이 떨어집니다.

#### Overfitting (과대적합) / High Variance (높은 분산)

- **개념:** 모델이 훈련 데이터에 너무 지나치게 맞추어져, 데이터의 노이즈(오차)나 미세한 변동까지 복잡하게 학습한 상태입니다.
    
- **이유:** 4차 다항식($x, x^2, x^3, x^4$)처럼 너무 많은 특성을 사용해 모델이 필요 이상으로 복잡해졌기 때문입니다.
    
- **이름의 원인 (High Variance):** 훈련 데이터가 조금만 바뀌어도 모델의 형태가 완전히 크게 뒤흔들리기 때문에 "분산(Variance)이 높다"고 표현합니다.
    
- **결과:** 훈련 데이터에 대해서는 오차 0에 가까운 완벽한 성능을 보이지만, **새로운 데이터가 들어오면 예측력이 현저히 떨어집니다 (일반화 실패).**

---
---
### 2. Addressing Overfitting
#### 0).과대 적합  해결 방법

- 모델에 과대적합(High Variance)이 발생했을 때 해결할 수 있는 방법은 크게 다음 3가지입니다.
	- 1.더 많은 데이터 수집하기(Collect More Data)
	- 2.특성 선택 (Feature Selection)
	- 3.정규화(Regularization)


#### 1). Collect More Training Data

- **원리:** 학습할 데이터의 수가 늘어나면, 모델이 4차 다항식 같은 고차원 모델(High-order polynomial)을 사용하더라도 데이터 전체의 흐름을 반영하느라 요동치는 현상(Wiggly curve)이 줄어듭니다.
    
- **장점:** 과대적합을 막는 가장 최고의 1순위 해결법(Number one tool)입니다.
    
- **한계:** 실제 환경에서는 추가 데이터를 구하기 어렵거나 불가능한 경우가 많습니다. (예: 특정 지역에서 팔린 집 데이터의 한계)
    

#### 2). Feature Selection

- **원리:** 너무 많은 특성(Feature)이 존재할 때, 예측에 중요하지 않거나 불필요한 특성을 제외하고 **가장 관련성이 높은 핵심 특성의 소수 집합(Subset)만 선택**하여 모델을 단순화합니다.
    
    - _예시:_ 100개의 특성($x_1, x_2, \dots, x_{100}$) 중 집값 예측에 가장 중요한 `집 크기(Size)`, `침실 수(Bedrooms)`, `연식(Age)`만 골라내어 사용.
        
- **단점:** 불필요해 보이는 특성을 버리는 과정에서 유용한 정보까지 함께 손실(Throw away information)될 위험이 있습니다.
    
- **참고:** 수동으로 특성을 골라낼 수도 있지만, 코스 2(Course 2)에서 적절한 특성을 자동으로 선택해 주는 알고리즘을 다루게 됩니다.
    

#### 3). Regularization

- **원리:** 특성을 아예 제거해 버리는 대신, 모든 특성을 유지하되 매개변수(가중치 $w_j$)의 크기를 작게 축소(Shrink)시킵니다.
    
    - 특성을 아예 삭제하는 것은 가중치를 $w_j = 0$으로 만드는 것과 같습니다.
        
    - 정규화는 $w_j$를 완전히 0으로 만들지 않고, **가중치의 영향을 완만하게 줄여 주는 역할**을 합니다.
        
- **특징:**
    - 모든 특성이 지닌 정보(Information)를 버리지 않고 유지할 수 있습니다.
        
    - 특정 특성이 모델의 예측 결과에 지나치게 거대한 영향을 미치는 것을 방지합니다.
        
    - **관례(Convention):** 주로 가중치 매개변수 $w_1, w_2, \dots, w_n$의 크기를 줄이는 데 집중하며, 편향(Bias) 매개변수 $b$는 정규화하든 안 하든 결과에 큰 차이가 없기 때문에 **보통 $b$는 정규화 대상에서 제외**합니다.

---
---
### 3. Cost Function with Regularization
- 정규화 시에 손실 함수에는 Regularization Term이라는 것이 붙는다.
#### 0). 정규화 비용 함수 (Regularized Cost Function)
정규화는 **비용 함수(Cost Function)에 가중치($w$)의 크기를 제약하는 '벌점(Penalty)' 항목을 추가**하여 모델의 과대적합(Overfitting)을 방지하는 기법입니다.

```math
J(\mathbf{w}, b) = \underbrace{\frac{1}{2m} \sum_{i=1}^{m} \left( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \right)^2}_{\text{기본 비용 함수 (Mean Squared Error)}} + \underbrace{\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2}_{\text{정규화 항목 (Regularization Term)}}
```

#### 1). 정규화의 기본 직관

- **아이디어:** 높은 차수의 복잡한 모델(예: 4차 다항식)이 있을 때, 특정 가중치($w_3, w_4$)를 거의 0에 가깝게 만들 수 있다면 모델은 훨씬 매끄럽고 단순해집니다.
    
- **패널티 부여:** 비용 함수에 가중치의 제곱 항목($w_3^2, w_4^2$)을 크게 더해주면, 알고리즘은 전체 비용 $J$를 줄이기 위해 스스로 $w_3, w_4$를 **0에 가까운 매우 작은 값**으로 축소시키게 됩니다.
	
- Regularization Term을 미분하면 아래와 같은 수식이 된다. 
 ```math
w_j \leftarrow \left(1 - \alpha \frac{\lambda}{m}\right) w_j - \alpha \times (\text{기존 오차 기울기})
```
- 매 단계(Iter)마다 가중치 $w_j$에 **$1$보다 작은 값(예: $0.99$)을 먼저 곱한 뒤**에 업데이트합니다. 
	
- 즉, 데이터가 가중치를 키우려고 아무리 당겨도, 수식 자체에 **"매번 가중치를 일정 비율 깎아먹는 장치"**가 내장되어 있어서 자연스럽게 $w$가 0을 향해 축소(Shrinkage)되는 것입니다.
---
#### 2). 전체 가중치 정규화 (Regularizing All Parameters)

실제 특성(Feature)이 100개 이상으로 많을 때는 어떤 특성을 줄여야 할지 사전에 알기 어렵습니다. 따라서 **모든 가중치 $w_1, \dots, w_n$ 전체에 일률적으로 벌점을 부여**합니다.

##### (1). 두 가지 목표의 균형 (Trade-off)

수정된 비용 함수는 다음 두 가지 목표 사이에서 균형을 잡습니다:

1. **첫 번째 항목 (MSE):** 훈련 데이터의 예측 오차를 최소화하여 **데이터에 잘 맞추는 것**
    
2. **두 번째 항목 (Regularization):** 가중치 $w_j$의 크기를 작게 유지하여 **모델을 매끄럽고 단순하게 만드는 것**
    

##### (2). 세부 규칙 및 관례

- **$\frac{1}{2m}$ 스케일링:** 정규화 항목도 기본 비용 함수와 동일하게 $2m$으로 나누어 줍니다. 이렇게 하면 **데이터 세트의 크기($m$)가 커지더라도 기존에 정해둔 $\lambda$ 값을 그대로 유지**하여 사용할 수 있습니다.
    
- **$b$(Bias) 제약외:** 관례적으로 편향 $b$는 정규화 항목에 포함하지 않습니다. (포함하더라도 성능에 유의미한 차이가 없음)
---
#### 3). 정규화 매개변수 Lambda의 역할

| **λ 값**                            | **모델 상태**               | **설명**                                                               |
| ---------------------------------- | ----------------------- | -------------------------------------------------------------------- |
| **$\lambda = 0$**                  | **Overfitting** (과대적합)  | 정규화 항목이 사라져 훈련 데이터에 너무 구불구불하게 들어맞음                                   |
| **$\lambda$가 매우 큼** (예: $10^{10}$) | **Underfitting** (과소적합) | 정규화 비중이 너무 커서 모든 $w_j \approx 0$이 됨. 결국 $f(x) \approx b$인 단순한 수평선이 됨 |
| **적절한 $\lambda$**                  | **Just Right** (바람직함)   | 복잡한 고차원 특성을 유지하면서도 예측 곡선이 유연하고 매끄럽게 일반화됨                             |

---
---
### 4. Regularized Logistic Regression
- 선형 회귀와 마찬가지로, **로지스틱 회귀(Logistic Regression)** 역시 복잡한 고차 다항식(High-order polynomial)이나 너무 많은 특성(Feature)을 사용할 경우 과대적합(Overfitting)이 발생하여 decision boundary가 지나치게 복잡해질 위험이 있습니다.
	
- 이를 해결하기 위해 로지스틱 회귀의 비용 함수에도 정규화 항목(Regularization Term)을 추가합니다.
#### 1). 정규화된 로지스틱 회귀의 비용 함수
```math
J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\left(f_{\mathbf{w},b}(\mathbf{x}^{(i)})\right) + (1 - y^{(i)}) \log\left(1 - f_{\mathbf{w},b}(\mathbf{x}^{(i)})\right) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
```
- **수학적 효과:** 가중치 $w_1, w_2, \dots, w_n$의 크기가 지나치게 커지는 것을 억제합니다.
    
- **결과:** 고차 다항식 특성을 사용하더라도, 훈련 데이터의 노이즈에 과도하게 반응하지 않는 매끄럽고 합리적인 결정 경계(Decision Boundary)를 형성하여 새로운 데이터에 대한 일반화(Generalization) 능력이 향상됩니다.

#### 2). 경사하강법 업데이트 수식
- 비용 함수를 최소화하기 위해 경사 하강법(Gradient Descent)을 사용할 때, 편미분(Derivative) 항에 $\frac{\lambda}{m} w_j$가 추가됩니다.
##### (1). 가중치 업데이트 수식
```math
w_j \leftarrow w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} w_j \right]
```

- 수식 형태는 선형 회귀와 **완벽히 동일**해 보이지만, 가중치 조합을 입력받는 함수 $f_{\mathbf{w},b}(\mathbf{x})$의 정의가 다릅니다.
	
- **선형 회귀:** $f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$
    
- **로지스틱 회귀:** $f_{\mathbf{w},b}(\mathbf{x}) = g(\mathbf{w} \cdot \mathbf{x} + b) \quad \left(\text{단, } g(z) = \frac{1}{1 + e^{-z}} \text{ Sigmoid 함수}\right)$

##### (2). 편향 업데이트 수식
```math
b \leftarrow b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \right)
```

- 편향 $b$는 정규화 항목에 포함되지 않으므로 기존 로지스틱 회귀 수식과 동일합니다.
