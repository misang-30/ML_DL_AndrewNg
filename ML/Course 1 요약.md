# Week1
- 선형회귀
## 1. 머신러닝 종류
- 지도 학습 (입력 데이터(0), 결과 데이터(0))
- 비지도 학습(입력 데이터(0), 결과 데이터(X))
- 추천 시스템
- 강화 학습

## 2. 선형 회귀
### 1). 가설 함수 설정
- f(x)=wx + b

### 2).  비용 함수 설정
- 평균 제곱 오차 사용
- 오차를 비용 함수로 가공한다.


$$\text{Error}^{(i)} = f_{w,b}(x^{(i)}) - y^{(i)} = \hat{y}^{(i)} - y^{(i)}$$

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$


### 3). 경사 하강법으로 파라미터 업데이트
$$w \leftarrow w - \alpha \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$
---
---
---
# Week2
- Multiple Regression, 실무팁
## 1. Multiple Regression
- 이것도 Linear Regression의 한 종류
### 0). 개념
- 입력 변수 2개 이상
- f w,b(x) = w1x1 + ... + w_n * x_n + b
- f w,b(x) = w벡터 * x벡터 + b (Dot Product) 

### 1). Dot 곱(내적)

- 두 벡터가 **얼마나 같은 방향을 향하고 있는지**를 측정하는 연산입니다.
- **기하학적 정의:**
    $$\mathbf{a} \cdot \mathbf{b} = \vert{}\mathbf{a}\vert{} \vert{}\mathbf{b}\vert{} \cos\theta$$
- **성분(Component) 정의:**    $$\mathbf{a} \cdot \mathbf{b} = a_x b_x + a_y b_y + a_z b_z$$

### 2). Cross 곱 (외적)
- 두 벡터가 이루는 평면에 **동시에 수직인 새로운 3차원 벡터**를 생성합니다.
-  **크기 (Magnitude):**
    $$\vert{}\mathbf{a} \times \mathbf{b}\vert{} = \vert{}\mathbf{a}\vert{} \vert{}\mathbf{b}\vert{} \sin\theta$$
    
    - 두 벡터가 만들어내는 **평행사변형의 넓이**와 같습니다.    

-  **방향 (Direction):**
    - **오른손 법칙**을 따릅니다.
    - 첫 번째 벡터($\mathbf{a}$)에서 두 번째 벡터($\mathbf{b}$) 방향으로 손가락을 감싸쥘 때, **엄지손가락이 가리키는 방향**입니다.

- **성분(Component) 정의:**  
$$\mathbf{a} \times \mathbf{b} = (a_y b_z - a_z b_y,\; a_z b_x - a_x b_z,\; a_x b_y - a_y b_x)$$

### 3). 벡터화
- 계산하나 하나 하기 힘드니까. 벡터로 묶어서 행렬 연산 하면 더 빠르다.
- Numpy의 np.dot을 쓰면 빠르다.
>f = np.dot(w,x) + b 


### 4). 비용 함수와 GD
- 비용 함수는 평균 제곱 오차를 사용한다.
- 다만, 가설 함수가 f_w,b(x) = w1x1 + w2x2 +... + wnxn + b  이다.
$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$
- n>=2 에서 비용 함수에 가설 함수 넣고 미분하고 GD에 적용하면 
- 아래와 같은 꼴이다. 미분은 w1,w2...wn에 대해 각각 한다.
  $$w_1 = w_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_1^{(i)}$$
  $$\vdots$$
  $$w_n = w_n - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_n^{(i)}$$

  $$b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)$$
### 5). GD의 대안 (Linear Regression 한정)
- Normal Equation(정규 방정식) 이라는 것이 있다.
- 반복 없이 w,b를 구할 수 있다. 공식을 통해 한 번에 최적값 산출.
- 공식은 따로 찾아볼 것.

---
---
## 2. Linear Regression 실무 팁
### 1). Feature Scaling 
- 서로 다른 특성들이 너무 다른 범위 값을 가지면, GD 느려진다.
- 특성들의 값을 조정하는 게 이럴 때 좋다.
	- 1. Scaling By Maximum : Feature의 최대값으로 나누어서 1 아래 값으로 스케일 하는 방법이 있다.
	- 2.Mean Normalization : 평균값으로 나누어서 Feature가 음수와 양수 범위에서 동작하게 할 수 있다.
	- 3.Z-Score Normalization : 표준편차 이용해서 하는 방법
### 2). GD가 수렴하는지 체크하라.
- 1.학습 곡선을 활용한다.
	- 학습 곡선  : GD 반복 횟수(X축) - 비용 함수 (Y축)
	- 매 반복마다 비용 J가 지속적으로 감소하면 제대로 작동하는 것.
	- 중간에 J가 증가하면 학습률이 너무 크거나 버그 있는 것.
- 2.자동 수렴 검사.
	- 한 번의 반복(Iteration) 동안 **비용 $J$의 감소량이 임계값보다 적으면** 수렴한 것으로 간주하고 학습을 종료합니다.
	- 입계값은 아주 작은 값으로 설정

### 3). 학습률 선택 방법
- 학습률이 너무 크면, 오버슈팅, 바운싱 현상 나온다.
- 학습률이 너무 작으면, 수렴하는데 많은 시간이 걸린다.
- GD가 잘안되는 경우 디버깅 : 
	- 학습률을 아주 작게 설정하고, 만약 비용 J가 매번 감소하지 않는다면 이것은 코드의 버그이다.  
- 학습률 선택 방법
	- 학습률 후보군 설정(3배수 법칙 ) : 0.001, 0.003,0.01,0.03...
	- 각 학습률의 GD반복횟수-비용함수 그래프 측정
	- 가장 좋은 것 선택.

### 4). 특성 공학
- 기존에 있는 특성을 가지고 변환하거나 조합해서 새로운 특성 만드는 것.

---
---
---
# Week3
- 분류(Classification) 다룸.
- 선형 회귀는 이상치에 민감하여, 분류 문제에는 적합하지 않다.
- 그래서 로지스틱 회귀를 사용한다.
## 1. 로지스틱 회귀
- 일단 이진 분류 용이다. (0,1 출력)
### 1). 모델 방정식
- 선형 회귀의 결과값을 시그모이드 함수에 넣은 꼴이다.
- 다항 회귀나 다른 모델을 시그모이드 함수 넣을 수도 있다.
 $$f_{\mathbf{w},b}(\mathbf{x}) = g(\mathbf{w} \cdot \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$$
- 결과 해석 : 입력이 종양 크기 일 때, 결과값 0.7은 이것이 악성일 확률 70%

### 2). 임계값
- 모델의 출력값이 0.5 이상이면 클래스1, 아니면 클래스 0
- 임계값 0.5 사용

### 3). 결정 경계(Decision Boundary)
- g(wx+b) = 0.5를 기준으로 클래스 판별을 하는데, 임계값은 0.5였다.
- 이때, wx+b = 0이다. 이를 결정 경계라고 부른다.
- 이 결정 경계는 선형일 수도 있고, 비선형 일 수도 있다.(예:타원형)

---
## 2. 비용 함수 
- 기존 "제곱 오차 손실 함수"를 로지스틱 회귀에 사용하면 Non-Convex 형태가 되어서 
- Global 최저점이 아니라, Local 최저점을 찾는 경사하강법이 수행될 수 있다.
- 이를 해결하기 위해
- 로지스틱 회귀에서는 다른 손실 함수 사용한다.

### 1). 손실 함수 (MLE, 최대 우도 추정)

- 실제 정답이 1인 경우와 실제 정답이 0인 경우에 따라 다르다.
> $$L(f(\mathbf{x}), y) = \begin{cases} > -\log(f(\mathbf{x})) & \text{if } y = 1 \\ > -\log(1 - f(\mathbf{x})) & \text{if } y = 0 > \end{cases}$$
$$L(f(\mathbf{x}), y) = -y \log(f(\mathbf{x})) - (1 - y) \log(1 - f(\mathbf{x}))$$

### 2). 비용함수

$$J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^{m} L(f(\mathbf{x}^{(i)}), y^{(i)})$$

---
## 3. 경사 하강법

- **가중치($w_j$) 업데이트:**
    $$w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$
    
- **편향($b$) 업데이트:**
    $$b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \right)$$
    

_(단, $\alpha$는 학습률(Learning Rate), $m$은 전체 데이터 개수)_


---

## 4. 규제 ( 오버 피팅 방지)

### 1). 오버 피팅의 문제점
- 언더피팅 / 높은 편향 : 데이터의 패턴조차 반영하지 못한 상태
- 오버피팅 / 높은 분산 : 데이터의 노이즈도 학습한 투머치 학습 상태

### 2). 오버 피팅 해결책
- 1.데이터 더 구해
- 2.특성 선택 ( 관련성 높은 것 위주로)
- 3.정규화 ( 특성은 유지하되, 필요없는 것은 가중치 줄여)
- 4.비용 함수에 정규화 적용( 오버 피팅 방지기법, 필요 없는 특성의 가중치를 낮춤 )
