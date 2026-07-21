- Week 1: Introduction to Machine Learning
### 0. ML 종류
- **Supervised Learning (지도 학습):** 정답($y$, Target)이 주어지는 학습 방식.
    
- **Unsupervised Learning (비지도 학습):** 정답 없이 오직 입력 데이터($x$)의 특징만으로 패턴을 찾는 방식.
    
- **Recommender Systems (추천 시스템):** 사용자 선호도를 예측하여 아이템을 제안하는 시스템 (주로 협업 필터링이나 콘텐츠 기반 필터링 사용).
    
- **Reinforcement Learning (강화 학습):** 에이전트가 환경과 상호작용하며 '보상(Reward)'을 극대화하는 행동 노선을 스스로 학습하는 방식.

---
---
---

### 1. Supervised Learning
> **핵심:** 입력 데이터($x$)와 결과 데이터($y$) 쌍으로 구성된 데이터셋을 주고, 둘 사이의 관계를 나타내는 함수 $f$를 학습시킵니다.

- **Regression (회귀):** 연속적인 숫자 값을 예측합니다. 출력값의 범위가 무한합니다. (예: 집값 예측, 주가 예측)
    
- **Classification (분류):** 이산적인 부류(Category)를 예측합니다. 출력값의 종류가 유한합니다. (예: 스팸 메일 여부 $[0, 1]$, 손글씨 숫자 $[0, 1, 2, \dots, 9]$)


---
---
---
### 2. Unsupervised Learning
> **핵심:** 데이터에 정답(Label)이 존재하지 않으며, 알고리즘이 데이터 자체의 구조나 패턴을 스스로 발견해야 합니다.

- **Clustering (군집화):** 유사한 특성을 가진 데이터끼리 스스로 그룹(Cluster)을 묶는 작업입니다. (예: 고객 세분화)
    
- **Dimensionality Reduction (차원 축소):** 정보 손실을 최소화하면서 시각화나 효율성을 위해 데이터의 변수(차원) 개수를 줄입니다. (예: PCA)
    
- **Association Rule Learning (연관 규칙 학습):** 항목 간의 숨겨진 동시 발생 관계를 찾습니다. (예: 장바구니 분석 - 기저귀와 맥주)
    
- **Anomaly Detection (이상 탐지):** 정상적인 패턴에서 벗어난 특이한 데이터를 찾아냅니다. (예: 카드 부정 사용 감지)
    
- **Generative Models (생성 모델):** 데이터의 분포를 학습하여 실제와 유사한 새로운 데이터를 생성합니다. (예: GAN, Diffusion)


---
---
---


## 3. Linear Regression Model
**개념:** 하나의 독립 변수(Feature) $x$와 종속 변수(Target) $y$의 선형적 관계를 모델링합니다. 변수가 하나이기 때문에 **'Linear Regression with one variable'** 혹은 **'Univariate Linear Regression'**이라고 부릅니다.

### 1). 데이터 표기법 (Notation)

- $x$: 입력 변수 (Feature, 특징)
    
- $y$: 실제 정답 변수 (Target, 대상)
    
- $\hat{y}$ ($y\text{-hat}$): 모델이 예측한 값 (Estimated $y$, Prediction)
    
- $m$: 학습 데이터셋의 총 개수 (Number of training examples)
    
- $(x^{(i)}, y^{(i)})$: $i$번째 데이터 샘플 ($i$는 지수가 아니라 몇 번째 샘플인지를 뜻함)

### 2). 워크플로우 (Workflow)

Training Set $\rightarrow$ Learning Algorithm $\rightarrow$ $f$ (Model)  

이후 새로운 $x$가 입력되면 $\hat{y} = f(x)$를 출력합니다.  

### 3). 가설 함수 (Hypothesis Function)

$$f_{w,b}(x) = wx + b$$

- $w$ (Weight, 가중치): 직선의 **기울기**를 결정합니다.
    
- $b$ (Bias, 편향): 직선의 **y절편**을 결정합니다.
    
- **목표:** 데이터셋에 가장 잘 맞는 최적의 매개변수(Parameter) $w$와 $b$를 찾는 것입니다.


---
---
---
## 4. Cost Function
> **목적:** 우리가 정한 $w$와 $b$가 실제 데이터에 얼마나 잘 맞는지(정확히는 얼마나 틀렸는지)를 수치화하여 평가합니다.

오차가 작을수록 좋은 모델이므로, 비용 함수의 값을 최소화하는 $w, b$를 구해야 합니다.

### 1). 오차 (Error) 정의

$i$번째 데이터에 대한 오차는 예측값과 실제값의 차이입니다.

$$\text{Error}^{(i)} = f_{w,b}(x^{(i)}) - y^{(i)} = \hat{y}^{(i)} - y^{(i)}$$

### 2). 비용 함수: 평균 제곱 오차 (Mean Squared Error, MSE)

회귀 분석에서 가장 흔하게 쓰이는 비용 함수이며, 오차를 제곱하여 평균을 냅니다. 기호로는 $J(w,b)$라고 씁니다.

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$


---
---
---
## 5. Cost Function 
- 비용함수는 작을 수록 좋다.
- J(w) (b=0 가정)에서 값이 가장 작은 값에서의 w를 구해야 한다.
- (J(w)는 w,b에 대한 그래프이다.)
- Contour Plot을 이용하면 w,b에 대한 J(w)를 보여주기 쉽다
- "경사하강법으로 w,b를 구한다."
- 

## 6. Train the model with GD
### 1). GD
- 초기값 w,b를 설정하고
- J(w,b)를 감소시키는 방법으로 계속 w,b를 바꾼다.
- 가장 높은 정상에서 가장 낮은 골로 가는 형상인데, 가장 가파른 곳의 방향으로 정상에서 내려온다.
> [!Notice]
>아래 식을 동시 업데이트해야 한다.  L이 같아야 한다. 서로 다르게 하면 달라진다.
>$$w \leftarrow w - \alpha \frac{\partial L}{\partial w}$$
>$$b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$


### 2). GD 직관
- 기울기가 양수이면 손실 함수를 w에대해 편미분한 값을 빼주는게 더 작은 값이 있는 곳으로 가는 방향이다.

### 3). Learning Rate (학습률)
- Alpha가 너무 작으면 GD는 매우 늘릴 것.
- Alpha가 너무 크면 최소에 가까운 값에서 진동할 것.(오버슈트)
- 만약에 w가 local minimum에 들어갔으면? 이것보다 작은 값이 있는데 현재 극소에 있으면 GD 식으로는 더 작은 값을 업데이트 할 수 없다.

####  (1). 제곱 오차 비용 함수의 볼록성 (Convex Nature)

일반적인 경사 하강법에서는 전체 최솟값인 전역 최솟값(Global Minimum)을 찾지 못하고, 지역적인 최솟값인 지역 최솟값(Local Minimum)에 갇히는 문제가 발생할 수 있습니다.

하지만 **선형 회귀의 제곱 오차 비용 함수**를 사용할 때는 매우 강력한 장점이 있습니다:

- 이 비용 함수는 볼록 함수 (Convex function)입니다. 쉽게 말해 '밥그릇 모양'의 형태를 띱니다.
    
- 볼록 함수는 여러 개의 지역 최솟값이 존재하지 않으며, 오직 **단 하나의 전역 최솟값**만 가집니다.
    

**결론:** 따라서 학습률($\alpha$)만 적절하게 잘 선택한다면, 선형 회귀에서 경사 하강법은 언제나 예외 없이 절대적인 최적의 결론(전역 최솟값)으로 수렴하게 됩니다.

### 4). Linear Regression 에서 GD 적용
- 아래가 공식이다. J(w)를 미분한 값을 넣어둔 것이다.
$$\begin{aligned} w &= w - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)} \\ b &= b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) \end{aligned}$$

### 5). Running GD 
- 우리가 지금까지 배운 경사 하강법 프로세스를 "배치 경사 하강법"이라 한다.
- **배치(Batch)의 의미:** 경사 하강법의 **매 단계(Step)마다 훈련 데이터의 일부가 아니라 '전체 훈련 데이터(전체 배치)'를 모두 확인**하여 미분과 업데이트를 계산한다는 뜻입니다.


---
---
---

