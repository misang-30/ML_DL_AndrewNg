

1주차는 비지도 학습(Unsupervised Learning)의 대표적 알고리즘인 **K-평균(K-means) 군집화**와 이상 탐지(Anomaly Detection)를 다룹니다.

## 1. 비지도 학습 (Unsupervised Learning)이란?

- **지도 학습(Supervised Learning):** 입력 데이터 $x$와 정답 라벨 $y$가 함께 제공됨.
    
- **비지도 학습(Unsupervised Learning):** 정답 라벨 $y$가 **없고** 입력 데이터 $x$만 존재함.
    
- **목적:** 데이터 내부의 숨겨진 구조, 패턴, 그룹을 스스로 찾아내는 것.
    

## 2. K-평균 군집화 (K-means Clustering)

데이터를 유사한 특징을 가진 $K$개의 그룹(Cluster)으로 묶어주는 대표적인 군집화 알고리즘입니다.

### ⚙️ 동작 과정

1. **중심점(Centroid) 초기화:** K개의 중심점을 임의의 위치에 배치.
    
2. **데이터 할당 (Assign):** 각 데이터 포인트를 가장 가까운 중심점에 할당.
    
3. **중심점 이동 (Move):** 각 클러스터에 속한 데이터들의 평균 위치로 중심점 이동.
    
4. **반복:** 중심점 위치가 더 이상 바뀌지 않을 때까지 2~3 과정을 수렴할 때까지 반복.
    

### 💡 주요 핵심 요소

- **비용 함수 (Cost Function / Distortion Function):**
    
    - 각 데이터와 소속된 클러스터 중심점 사이의 거리 제곱합을 최소화하는 것이 목적.
        
- **중심점 초기화 문제 (Random Initialization):**
    
    - 초기 중심 위치에 따라 지역 최적값(Local Optima)에 갇힐 수 있음.
        
    - **해결책:** K-means를 여러 번(예: 50~100회) 무작위 초기화하여 run한 후, 비용 함수 $J$가 가장 작은 결과를 선택.
        
- **클러스터 개수 $K$ 선택하기:**
    
    - **Elbow Method:** $K$를 늘려가며 비용 함수 감소폭이 꺾이는 지점("팔꿈치")을 선택.
        
    - **목적 기반 선택 (Downstream Purpose):** 비즈니스 목적(예: 의류 사이즈 S/M/L 3개 구분 등)에 따라 직접 지정.
        

## 3. 이상 탐지 (Anomaly Detection)

정상적인 데이터 패턴을 학습한 뒤, 새로운 데이터가 들어왔을 때 정상 범주를 벗어난 희귀한 데이터(이상치)를 찾아내는 기술입니다.

### ⚙️ 가우시안 분포 (정규분포) 모델링

- 각 특징(Feature) $x_i$가 평균 $\mu_i$, 분산 $\sigma_i^2$을 갖는 가우시안 분포(Gaussian Distribution)를 따른다고 가정합니다.
    
- 새로운 데이터 $x$가 나타날 확률 density $p(x)$를 아래와 같이 계산합니다:
    
    $$p(x) = p(x_1; \mu_1, \sigma_1^2) \times p(x_2; \mu_2, \sigma_2^2) \times \dots \times p(x_n; \mu_n, \sigma_n^2)$$
    
- **판정 기준:**
    
    - $p(x) < \epsilon$ (임계값): 이상치 (Anomaly)로 판정
        
    - $p(x) \ge \epsilon$: 정상 (Normal)으로 판정
        

### ⚖️ 이상 탐지 vs 지도 학습 (분류)

|**구분**|**이상 탐지 (Anomaly Detection)**|**지도 학습 (Classification)**|
|---|---|---|
|**양성(Positive) 데이터 수**|매우 적음 (0~20개 내외의 이상 데이터)|양성/음성 데이터 모두 충분히 많음|
|**이상 유형**|수많은 **알 수 없는 새로운 유형**의 불량이 발생할 수 있음|기존 데이터와 **유사한 유형**의 패턴을 계속 분류함|
|**주요 사용 예시**|사기 검출(Fraud), 서버 감시, 제조 결함 탐지|스팸 메일 분류, 암 진단, Weather 예측|

> 💡 **1주차 핵심 요약**
> 
> - **K-means:** 라벨 없는 데이터를 거리 기반으로 $K$개 그룹으로 묶는 알고리즘.
>     
> - **Anomaly Detection:** 정상 데이터의 가우시안 확률 분포 $p(x)$를 만들어, 확률이 임계값 $\epsilon$보다 낮으면 이상 데이터로 분류하는 기법.



---
---
# 자세한 버전
## Part 1. K-평균 군집화 (K-means Clustering)

### 1-1. 지도 학습 vs 비지도 학습

- **지도 학습(Supervised Learning):** 학습 데이터셋 $\{ (x^{(1)}, y^{(1)}), \dots, (x^{(m)}, y^{(m)}) \}$
    
- **비지도 학습(Unsupervised Learning):** 학습 데이터셋 $\{ x^{(1)}, x^{(2)}, \dots, x^{(m)} \}$ (라벨 $y$ 없음)
    

### 1-2. K-means 알고리즘 최적화 과정

1. **클러스터 중심점(Centroid) 무작위 초기화:** $K$개의 중심점 $\mu_1, \mu_2, \dots, \mu_K \in \mathbb{R}^n$ 선택
    
2. **Loop 수렴할 때까지 반복:**
    
    - **[Step 1] 데이터 할당 (Cluster Assignment):**
        
        각 데이터 $i=1 \dots m$에 대해 가장 가까운 중심점의 인덱스 $c^{(i)}$ 계산
        
        $$c^{(i)} = \min_k \vert{}\vert{} x^{(i)} - \mu_k \vert{}\vert{}^2$$
        
    - **[Step 2] 중심점 이동 (Move Centroid):**
        
        각 클러스터 $k=1 \dots K$에 대해 할당된 데이터들의 평균으로 위치 업데이트
        
        $$\mu_k = \frac{1}{\vert{}c_k\vert{}} \sum_{i \in c_k} x^{(i)}$$
        
        _(단, 클러스터에 할당된 데이터가 0개면 해당 중심점은 삭제하거나 재초기화)_
        

### 1-3. 비용 함수 (Cost Function / Distortion Function)

K-means는 다음 비용 함수 $J$를 **최소화**하는 과정입니다:

$$J(c^{(1)}, \dots, c^{(m)}, \mu_1, \dots, \mu_K) = \frac{1}{m} \sum_{i=1}^{m} \vert{}\vert{} x^{(i)} - \mu_{c^{(i)}} \vert{}\vert{}^2$$

- Step 1은 $\mu$를 고정하고 $c^{(i)}$에 대해 $J$를 최소화.
    
- Step 2는 $c^{(i)}$를 고정하고 $\mu$에 대해 $J$를 최소화.
    

### 1-4. 국소 최적해(Local Optima) 탈출법

- 초기 중심점 선택에 따라 Global Optima가 아닌 Local Optima에 갇힐 수 있음.
    
- **Random Initialization 기법:**
    
    1. $K < m$ 인 데이터 포인트 $K$개를 무작위 추출하여 초기 중심점으로 설정.
        
    2. K-means를 **50~100번 독립적으로 실행**.
        
    3. 계산된 100개의 비용 함수 $J$ 중 **가장 값이 작은 결과**를 최종 모델로 선택.
        

### 1-5. 클러스터 개수 $K$ 결정 방법

1. **Elbow Method:** $K$에 따른 $J$의 감소 그래프를 그려 꺾이는 지점(Elbow) 선택. (단, 완만한 곡선일 경우 명확한 지점을 찾기 힘듦)
    
2. **후속 목적 기준 (Downstream Purpose):** T셔츠 사이즈 제작(S, M, L $\rightarrow K=3$) 등 비즈니스 요구사항에 따라 결정.
    

## Part 2. 이상 탐지 (Anomaly Detection)

### 2-1. 가우시안(정규) 분포 (Gaussian Distribution)

데이터 $x \in \mathbb{R}$가 평균 $\mu$, 분산 $\sigma^2$을 가질 때:

$$x \sim \mathcal{N}(\mu, \sigma^2) \implies p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right)$$

- **모수 추정 (Parameter Estimation):**
    
    $$\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}, \quad \sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2$$
    

### 2-2. 이상 탐지 알고리즘 절차

1. **특징 선택 (Feature Selection):** 정규분포를 잘 따르는 $n$개의 피처 $x_1, \dots, x_n$ 선정.
    
2. **모수 학습:** $m$개 데이터로 각 피처의 $\mu_1, \dots, \mu_n$ 및 $\sigma_1^2, \dots, \sigma_n^2$ 계산.
    
3. **확률 계산 (독립 가정):**
    
    $$p(x) = \prod_{j=1}^{n} p(x_j; \mu_j, \sigma_j^2) = \prod_{j=1}^{n} \frac{1}{\sqrt{2\pi}\sigma_j} \exp\left( -\frac{(x_j-\mu_j)^2}{2\sigma_j^2} \right)$$
    
4. **판정:** $p(x) < \epsilon$ 이면 **Anomaly (1)**, $p(x) \ge \epsilon$ 이면 **Normal (0)**.
    

### 2-3. 데이터 세트 분할 및 평가 방법

- **데이터 구성 예시:** 정상 데이터 10,000개 + 이상 데이터 20개
    
    - **Training Set:** 정상 6,000개 ($p(x)$ 파라미터 $\mu, \sigma^2$ 학습용, 라벨 없음)
        
    - **Cross Validation (CV) Set:** 정상 2,000개 + 이상 10개 ($\epsilon$ 최적화용)
        
    - **Test Set:** 정상 2,000개 + 이상 10개 (최종 성능 평가용)
        
- **평가 지표:** 데이터 불균형이 극심하므로 정확도(Accuracy) 대신 **Precision, Recall, $F_1$-Score** 사용.
    

### 2-4. Anomaly Detection vs Supervised Learning 비교

|**비교 항목**|**Anomaly Detection**|**Supervised Classification**|
|---|---|---|
|**Positive 데이터($y=1$) 수**|매우 적음 (0~20개)|많음 (양성/음성 모두 충분)|
|**Negative 데이터($y=0$) 수**|매우 많음|많음|
|**이상 패턴의 다양성**|알려지지 않은 새로운 형태의 이상 발생|기존 데이터를 닮은 알려진 패턴 판별|

### 2-5. 피처 엔지니어링 (Feature Engineering)

- **비가우시안 데이터 다루기:** 데이터가 종 모양이 아니면 $\log(x)$, $\log(x+c)$, $\sqrt{x}$, $x^{0.5}$ 변환을 취해 정규분포 형태로 변환.
    
- **새로운 피처 생성:** 정상 데이터인데 $p(x)$가 크게 나오는 이상치가 있다면, 두 피처의 비율(예: $\frac{\text{CPU load}}{\text{network traffic}}$) 등을 새 피처로 추가하여 구분력을 높임.

---
---


## 1. 코스 3 - 1주차: 비지도 학습 & 이상 탐지

### 1). K-평균 군집화 (K-means Clustering)

#### (1). 알고리즘 최적화 과정 및 비용 함수

- **알고리즘 절차:**
    
    1. $K$개의 중심점(Centroid) $\mu_1, \dots, \mu_K$를 무작위로 초기화.
        
    2. **[데이터 할당]** 각 데이터 $x^{(i)}$에 대해 가장 가까운 중심점 인덱스 $c^{(i)}$ 할당:
        
        $$c^{(i)} = \min_k \vert{}\vert{} x^{(i)} - \mu_k \vert{}\vert{}^2$$
        
    3. **[중심점 이동]** 각 클러스터에 할당된 데이터들의 평균 위치로 중심점 업데이트:
        
        $$\mu_k = \frac{1}{\vert{}c_k\vert{}} \sum_{i \in c_k} x^{(i)}$$
        
    4. 중심점 위치 변화가 없을 때까지 2~3 과정 반복.
        
- **비용 함수 (Distortion Function):**
    
    $$J(c^{(1)}, \dots, c^{(m)}, \mu_1, \dots, \mu_K) = \frac{1}{m} \sum_{i=1}^{m} \vert{}\vert{} x^{(i)} - \mu_{c^{(i)}} \vert{}\vert{}^2$$
    

#### (2). 국소 최적해 탈출 및 $K$ 선택 기법

- **Random Initialization:** 초기 중심 위치에 따라 Local Optima에 갇히는 문제를 막기 위해, K-means를 50~100회 독립적으로 실행한 뒤 비용 함수 $J$가 가장 작은 결과를 최종 선택.
    
- **Elbow Method:** $K$ 증가에 따른 $J$의 감소폭이 급격히 꺾이는 "팔꿈치" 지점을 선택하거나, 비즈니스 목적(예: 의류 사이즈 S/M/L)에 맞춰 직접 지정.
    

### 2). 이상 탐지 (Anomaly Detection)

#### (1). 가우시안 분포 모델링 및 판정

- 각 피처 $x_j$가 정규분포를 따른다고 가정하고 평균 $\mu_j$와 분산 $\sigma_j^2$을 계산.
    
- 전체 확률 밀도 함수 계산:
    
    $$p(x) = \prod_{j=1}^{n} p(x_j; \mu_j, \sigma_j^2) = \prod_{j=1}^{n} \frac{1}{\sqrt{2\pi}\sigma_j} \exp\left( -\frac{(x_j-\mu_j)^2}{2\sigma_j^2} \right)$$
    
- **판정 기준:** $p(x) < \epsilon$ 이면 이상치(Anomaly, 1), $p(x) \ge \epsilon$ 이면 정상(Normal, 0).
    

#### (2). 데이터 분할 및 평가 지표

- **데이터 구성:** 정상 데이터 위주로 Training Set(정상만으로 $\mu, \sigma^2$ 학습)을 구성하고, CV/Test Set에는 희귀한 이상치를 일부 포함시킴.
    
- **평가 지표:** 극심한 데이터 불균형 문제로 인해 Accuracy 대신 **Precision, Recall, $F_1$-Score**로 성능을 평가함.
    
- **피처 엔지니어링:** 비대칭 데이터는 $\log(x)$ 변환을 취해 가우시안 형태로 만들고, 필요한 경우 두 피처의 비율(예: $\frac{\text{CPU load}}{\text{network traffic}}$) 등을 신규 피처로 만듦.