
2주차는 넷플릭스, 유튜브, 아마존 등에서 필수적으로 쓰이는 추천 시스템(Recommender Systems)을 다루며, 핵심 방식인 **협업 필터링**과 **콘텐츠 기반 필터링**을 공부하게 됩니다.

## 1. 추천 시스템의 기본 개념

추천 시스템의 목표는 사용자 $u$가 아직 평가하지 않은 아이템 $i$에 대해 **어떤 평점 $y^{(u,i)}$을 줄지 예측**하는 것입니다.

- $r(i,j) = 1$: 사용자 $j$가 아이템 $i$에 평점을 남긴 경우
    
- $y^{(i,j)}$: 사용자 $j$가 아이템 $i$에 부여한 실제 평점 (예: 1~5점)
    

## 2. 협업 필터링 (Collaborative Filtering)

사용자들의 과거 행동 데이터(평점, 구매 이력)만을 활용하여 비슷한 취향을 가진 사용자를 찾아 추천하는 방식입니다.

### ⚙️ 동작 원리: 인수 분해 (Matrix Factorization)

각 아이템 $i$의 특징 벡터 $w^{(i)}$와 각 사용자 $j$의 선호도 벡터 $x^{(j)}$를 동시에 학습합니다.

- **예측 평점:** $w^{(i)} \cdot x^{(j)} + b^{(j)}$
    
- 아이템의 특징(예: 영화의 장르 비율)과 사용자의 성향을 따로 사람이 입력하지 않아도, **평점 데이터만 가지고 경사 하강법(Gradient Descent)을 통해 두 벡터를 동시에 자동으로 학습**해냅니다.
    

### 💡 평균 정규화 (Mean Normalization)

- **문제점:** 평점을 단 하나도 남기지 않은 신규 유저가 오면 예측 평점이 모두 0으로 계산됨.
    
- **해결책:** 각 영화의 평균 평점을 구해 모든 평점에서 뺀 뒤 모델을 학습시킵니다. 신규 유저에게는 단순히 **해당 영화의 전체 평균 평점**을 예측값으로 제공하게 됩니다.
    

## 3. 콘텐츠 기반 필터링 (Content-based Filtering)

아이템의 속성(장르, 배우 등)과 사용자의 속성(나이, 선호 장르 등)을 입력 피처로 받아 **딥러닝** 모델을 통해 추천하는 최신 방식입니다.

```
[사용자 특징 X_u] ───>  [User 신경망]   ───> Vector V_u (예: 128차원)
                                                 │
                                                 ├───> Dot Product (내적) ───> 예측 평점
                                                 │
[아이템 특징 X_i] ───> [Item 신경망]  ───> Vector V_i (예: 128차원)
```

1. **사용자 네트워크**와 **아이템 네트워크** 두 개의 신경망을 각각 구성합니다.
    
2. 각 네트워크는 임의 차원의 임베딩 벡터 $V_u$, $V_i$를 출력합니다.
    
3. 두 벡터의 내적($V_u \cdot V_i$)이 클수록 사용자가 해당 아이템을 좋아할 확률이 높다고 판단합니다.
    

## 4. 대규모 추천 시스템 구조 (Retrieval & Ranking)

실제 수천만 개의 아이템이 있는 서비스(예: 유튜브)에서는 모든 아이템의 점수를 일일이 계산하기 어렵습니다. 따라서 2단계 파이프라인을 사용합니다.

1. **후보 추출 (Retrieval Step):**
    
    - 유저가 최근 본 영화 10개와 유사한 영화, 가장 인기 있는 장르 등 빠르게 수백 개(100~500개)의 후보만 추려냄.
        
2. **순위 매기기 (Ranking Step):**
    
    - 학습시킨 콘텐츠 기반 딥러닝 모델에 추출된 후보들만 넣어서 정밀한 예측 점수 계산 후, 상위 N개를 유저에게 노출.
        

> 💡 **2주차 핵심 요약**
> 
> - **Collaborative Filtering:** 평점 행렬만으로 사용자 성향과 아이템 특징을 _동시에 자동 학습_하는 기법.
>     
> - **Content-based Filtering:** 딥러닝 신경망을 통해 유저 벡터 $V_u$와 아이템 벡터 $V_i$를 만들어 *유사도(내적)*를 계산하는 기법.
>     
> - **Retrieval + Ranking:** 대용량 추천 서비스에서 속도와 정확도를 모두 잡기 위한 2단계 추천 기법.


---
---
# 자세한 버전
## Part 1. 협업 필터링 (Collaborative Filtering)

### 1-1. 기호 및 명세

- $n_u$: 사용자(User) 수, $n_m$: 아이템(Item) 수
    
- $r(i,j) = 1$: 사용자 $j$가 아이템 $i$에 평가를 남긴 경우
    
- $y^{(i,j)}$: 사용자 $j$가 아이템 $i$에 준 실제 평점
    
- $w^{(j)}, b^{(j)}$: 사용자 $j$의 파라미터
    
- $x^{(i)}$: 아이템 $i$의 특징 벡터 (차원 $n$)
    

### 1-2. 협업 필터링의 핵심 알고리즘

기존에는 $x^{(i)}$를 사람이 직접 입력했으나, 협업 필터링은 **$x^{(i)}$와 $w^{(j)}, b^{(j)}$를 동시에 학습**합니다.

- **통합 비용 함수 (Combined Cost Function):**
    
    $$J(w, b, x) = \frac{1}{2} \sum_{(i,j):r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2$$
    
- **경사 하강법 (Gradient Descent) 업데이트:**
    
    $$x_k^{(i)} := x_k^{(i)} - \alpha \left( \sum_{j:r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right) w_k^{(j)} + \lambda x_k^{(i)} \right)$$
    
    $$w_k^{(j)} := w_k^{(j)} - \alpha \left( \sum_{i:r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right) x_k^{(i)} + \lambda w_k^{(j)} \right)$$
    
    $$b^{(j)} := b^{(j)} - \alpha \left( \sum_{i:r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right) \right)$$
    

### 1-3. 평균 정규화 (Mean Normalization)

- **Cold Start 문제:** 평점을 하나도 안 남긴 유저는 모든 $w, b$가 0이 되어 예측값이 모두 0이 됨.
    
- **해결책:**
    
    1. 각 아이템 $i$의 평균 평점 $\mu_i$ 계산.
        
    2. 평점 행렬 $Y$의 각 요소에서 $\mu_i$를 뺀 $\mathbf{Y}_{norm}$ 생성.
        
    3. $\mathbf{Y}_{norm}$으로 모델 학습 후, 최종 예측 시 다시 $\mu_i$를 더함:
        
        $$\text{Prediction} = w^{(j)} \cdot x^{(i)} + b^{(j)} + \mu_i$$
        

## Part 2. 콘텐츠 기반 필터링 (Content-based Filtering)

### 2-1. Deep Learning 기반 임베딩 방식

- **입력 데이터:**
    
    - 사용자 피처 $x_u$ (나이, 성별, 과거 선호 장르 등)
        
    - 아이템 피처 $x_i$ (장르, 개봉일, 배우 등)
        
- **네트워크 구조:**
    
    - User Network: $x_u \rightarrow v_u \in \mathbb{R}^k$ (유저 임베딩 벡터)
        
    - Item Network: $x_i \rightarrow v_i \in \mathbb{R}^k$ (아이템 임베딩 벡터)
        
- **예측 점수:**
    
    $$v_u \cdot v_i \quad \text{또는 Cosine Similarity} \quad \frac{v_u \cdot v_i}{\vert{}\vert{}v_u\vert{}\vert{} \vert{}\vert{}v_i\vert{}\vert{}}$$
    

### 2-2. 손실 함수 (Loss Function)

실제 평점이 $y^{(u,i)}$일 때:

$$J = \sum_{(u,i):r(u,i)=1} \left( v_u \cdot v_i - y^{(u,i)} \right)^2 + \text{Regularization}$$

## Part 3. 이진 반응 데이터 & 대규모 시스템 (Retrieval & Ranking)

### 3-1. Binary Labels (좋아요/클릭/구매)

- 평점 숫자가 아닌 1(관심/클릭)과 0(무관심/미클릭) 데이터 처리.
    
- 손실 함수로 **Binary Cross-Entropy** 사용:
    
    $$\text{Cost} = - y \log(f) - (1-y) \log(1-f) \quad \left(\text{단, } f = \sigma(v_u \cdot v_i + b)\right)$$
    

### 3-2. Retrieval & Ranking 파이프라인

수백만 개의 아이템을 실시간 처리하기 위한 구조:

1. **Retrieval (후보군 추출):**
    
    - 유저가 최근 시청한 아이템 10개와 비슷한 아이템
        
    - 유저의 선호 카테고리 Top 3 내 최신/인기 아이템
        
    - 대략 수백 개(100~500개)로 후보군 축소 (빠른 연산)
        
2. **Ranking (순위 매기기):**
    
    - Retrieval에서 뽑힌 수백 개 아이템에 대해서만 Content-based Deep Learning 모델 적용.
        
    - 정확한 $v_u \cdot v_i$ 점수 산출 및 내림차순 정렬 후 Top N개 노출.
        

이제 강의의 **공식, 기호, 세부 하이퍼파라미터 알고리즘 단계**까지 빠짐없이 포함되었습니다! 추가로 3주차 내용도 필요하시면 편하게 말씀하세요.