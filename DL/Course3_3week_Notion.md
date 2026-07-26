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