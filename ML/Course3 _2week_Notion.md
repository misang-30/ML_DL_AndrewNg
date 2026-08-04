
# 추천 시스템 (Recommender Systems)

추천 시스템은 특정 기준에 따라 예측치를 정렬(Rank)하는 머신러닝의 한 분야입니다. 일반적인 머신러닝 데이터셋과 달리, 사용자(User)나 상품(Product)과 같은 고유 식별자(ID) 기반으로 데이터를 다루어 음악, 뉴스, 상품 등을 개인 맞춤형으로 제공하는 데 사용됩니다.

추천 시스템의 두 가지 핵심 알고리즘은 협업 필터링(Collaborative Filtering)과 콘텐츠 기반 필터링(Content-based Filtering)입니다.

---

## 1. 기본 개념 및 표기법

추천 시스템의 기본 데이터 구조는 $n_i \times n_j$ 행렬 형태를 갖습니다.

* $n_i$: 아이템(Item)의 개수
* $n_j$: 사용자(User)의 개수
* $y^{(i, j)}$: 사용자 $j$가 아이템 $i$에 부여한 실제 평점(Numerical Rating)
* $r(i, j)$: 사용자 $j$가 아이템 $i$에 평점을 남겼으면 $1$, 데이터가 없으면(Null/Undefined) $0$


---

## 2. 협업 필터링 (Collaborative Filtering)

사용자들의 평가 이력을 비교하여, 유사한 사용자들의 선호도를 기반으로 아이템을 추천하는 방식입니다.

### 1) 비용 함수 (Cost Function)

선형 회귀(Linear Regression) 기반으로 예측 평점 $\hat{y}^{(i, j)} = w^{(j)} \cdot x^{(i)} + b^{(j)}$를 모델링합니다.

* **단일 사용자에 대한 비용 함수:**

$$J(w^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i: r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{k=1}^{n} (w_k^{(j)})^2$$


* **전체 사용자에 대한 비용 함수:**

$$J(w, b) = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i: r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2$$



### 2) 특성(Feature) 역공학 및 동시 학습

아이템의 특성 벡터 $x^{(i)}$가 주어지지 않은 경우, 파라미터 $w, b$와 특성 벡터 $x$를 **동시에 학습**할 수 있습니다.

* **동시 학습 비용 함수 (Combined Cost Function):**

$$J(w, b, x) = \frac{1}{2} \sum_{(i,j): r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2$$



### 3) 경사 하강법 (Gradient Descent)

비용 함수를 최소화하기 위해 $w, b, x$ 세 개의 변수를 동시에 업데이트합니다.

* $w_i^{(j)} := w_i^{(j)} - \alpha \frac{\partial}{\partial w_i^{(j)}} J(w, b, x)$
* $b^{(j)} := b^{(j)} - \alpha \frac{\partial}{\partial b^{(j)}} J(w, b, x)$
* $x_k^{(i)} := x_k^{(i)} - \alpha \frac{\partial}{\partial x_k^{(i)}} J(w, b, x)$

---

## 3. 이진 라벨(Binary Labels)로의 확장

좋아요/클릭 여부($1$ 또는 $0$)와 같은 이진 반응 데이터에는 로지스틱 회귀(Sigmoid)를 적용합니다.

* **예측 확률 모델:**

$$f_{(w,b,x)}(x) = g(w^{(j)} \cdot x^{(i)} + b^{(j)}) = \frac{1}{1 + e^{-(w^{(j)} \cdot x^{(i)} + b^{(j)})}}$$


* **단일 데이터 손실 함수 (Loss Function):**

$$L(f(x), y^{(i,j)}) = -y^{(i,j)} \log(f(x)) - (1 - y^{(i,j)}) \log(1 - f(x))$$


* **전체 이진 라벨 비용 함수:**

$$J(w, b, x) = \sum_{(i,j): r(i,j)=1} L(f(x), y^{(i,j)})$$



---

## 4. 평균 정규화 (Mean Normalization)

신규 사용자(Cold Start)처럼 평점을 전혀 남기지 않은 사용자의 경우, 예측값이 모두 $0$이 되는 문제를 해결하기 위해 사용합니다.

1. 각 아이템 $i$의 평균 평점 $\mu_i$를 계산합니다. (평점이 없는 데이터는 평균 계산에서 제외)


2. 기존 평점 행렬에서 평균을 뺀 $y^{(i,j)} - \mu_i$로 데이터를 정규화합니다.


3. 모델 예측 시 계산된 평균값 $\mu_i$를 다시 더해줍니다:



$$f(x) = w^{(j)} \cdot x^{(i)} + b^{(j)} + \mu_i$$



* **효과:** 데이터가 없는 신규 사용자에게 해당 아이템의 평균 평점을 기본 추천값으로 제공할 수 있으며, 학습 알고리즘의 수렴 속도를 향상시킵니다.



---

## 5. 유사 아이템 탐색 (Finding Related Items)

학습된 특성 벡터 $x^{(i)}$ 간의 유클리드 거리를 측정하여 유사한 아이템을 찾아 추천합니다.

$$\text{distance} = \sum_{l=1}^{n} (x_l^{(k)} - x_l^{(i)})^2 = \Vert{}x^{(k)} - x^{(i)}\Vert{}^2$$

---

## 6. 콘텐츠 기반 필터링 (Content-based Filtering)

아이템의 특성($x_m$)과 사용자의 프로필/특성($x_n$) 정보를 직접 신경망(Neural Network)의 입력으로 활용하여 매칭합니다.

### 1) 딥러닝 구조 (Two-Tower Network)

* **사용자 신경망 (User NN):** 사용자 특성 $x_n$을 입력받아 고정된 크기의 임베딩 벡터 $v_n^{(j)}$ 출력


* **아이템 신경망 (Item NN):** 아이템 특성 $x_m$을 입력받아 고정된 크기의 임베딩 벡터 $v_m^{(i)}$ 출력


* **예측 및 내적:** 두 임베딩 벡터의 내적(Dot Product)을 통해 선호도 예측:



$$\hat{y}^{(i,j)} = v_n^{(j)} \cdot v_m^{(i)}$$



### 2) 비용 함수

$$J = \sum_{(i,j): r(i,j)=1} \left( v_n^{(j)} \cdot v_m^{(i)} - y^{(i, j)} \right)^2 + \text{NN Regularization Term}$$

---

## 7. TensorFlow 구현 코드 예제

### 1) `tf.GradientTape`를 활용한 자동 미분 및 최적화

```python
import tensorflow as tf
from tensorflow import keras

# 1. 단일 변수 오토 디프(Auto-Differentiation) 예제
w = tf.Variable(3.0)
x = 1.0
y = 1.0  # target value
alpha = 0.01

iterations = 30
for iter in range(iterations):
    with tf.GradientTape() as tape:
        f_wb = w * x
        cost = (f_wb - y) ** 2

    # w에 대한 경사도(Gradient) 자동 계산
    [dJdw] = tape.gradient(cost, [w])
    w.assign_add(-alpha * dJdw)

# 2. 협업 필터링 알고리즘 루프 예제
optimizer = keras.optimizers.Adam(learning_rate=1e-1)
iterations = 200

for iter in range(iterations):
    with tf.GradientTape() as tape:
        cost_value = cofiCostFuncV(X, W, b, Ynorm, R, n, m, lambda_)

    # X, W, b 파라미터에 대한 경사도 추출 및 적용
    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))

```

### 2) TensorFlow를 활용한 콘텐츠 기반 필터링 (Two-Tower 모델)

```python
import tensorflow as tf
from tensorflow.keras import Model

# 사용자 신경망 정의
user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32)
])

# 아이템 신경망 정의
item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32)
])

# 사용자 입력 및 L2 정규화
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# 아이템 입력 및 L2 정규화
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# 두 벡터 간 유사도(내적) 계산
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# 최종 모델 및 손실함수 설정
model = Model(inputs=[input_user, input_item], outputs=output)
cost_fn = tf.keras.losses.MeanSquaredError()

```
