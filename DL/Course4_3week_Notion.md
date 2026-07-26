
Week 2에서는 컴퓨터 비전 분야의 역사적 이정표가 된 대표적인 CNN 모델들(LeNet, AlexNet, VGG)과, 층이 깊어질 때 발생하는 기울기 소멸 문제를 해결한 **ResNet**, 연산량을 비약적으로 줄인 **Inception / MobileNet / EfficientNet** 등 핵심 네트워크 구조를 다룹니다.

## 1. 고전적인 네트워크 (Classic Networks)

### **1) LeNet-5 (Yann LeCun, 1998)**

- **목적:** 손글씨 숫자(MNIST) 인식.
    
- **구조:** `Input (32x32) -> CONV -> POOL -> CONV -> POOL -> FC -> FC -> Output`
    
- **특징:**
    
    - 당시에는 Max Pooling 대신 Average Pooling을 사용했습니다.
        
    - 네트워크 뒤로 갈수록 공간적 크기($n_H, n_W$)는 줄어들고, 채널 수($n_C$)는 늘어나는 현대 CNN의 기본 패턴을 정립했습니다.
        

### **2) AlexNet (Alex Krizhevsky et al., 2012)**

- **목적:** ImageNet 이미지 분류 대회(ILSVRC) 우승.
    
- **구조:** `Input (227x227x3) -> CONV(11x11, s=4) -> POOL(3x3) -> ... -> FC -> FC -> Softmax(1000)`
    
- **특징:**
    
    - Sigmoid 대신 **ReLU**를 활성화 함수로 도입하여 학습 속도를 획기적으로 개선했습니다.
        
    - 과적합 방지를 위해 **Dropout**과 **Data Augmentation**을 적극 활용했습니다.
        
    - 약 6천만 개의 파라미터를 가졌으며, 당시 메모리 한계로 2개의 GPU로 나누어 병렬 연산을 수행했습니다.
        

### **3) VGG-16 (Simonyan & Zisserman, 2014)**

- **핵심 철학:** 복잡한 필터 크기 대신 **$3 \times 3$ CONV (stride=1, same) + $2 \times 2$ Max POOL (stride=2)** 규칙만 매우 단순하게 반복 적용.
    
- **특징:**
    
    - $3 \times 3$ 필터 2개를 직렬 배치하면 $5 \times 5$ 필터 1개와 동일한 수용장(Receptive Field)을 가지면서 파라미터 수는 대폭 줄어듭니다.
        
    - 파라미터 수가 약 1억 3,800만 개로 매우 많아 메모리 사용량이 큰 단점이 있습니다.
        

## 2. 잔차 네트워크 (ResNet - Residual Networks)

네트워크의 깊이($L$)가 깊어질수록 기울기 소멸/폭발(Vanishing/Exploding Gradients) 문제로 인해 오히려 학습 오차가 커지는 degradation 현상이 발생합니다. ResNet은 이를 Skip Connection (Short-cut)으로 해결했습니다.

### **Residual Block 연산**

일반적인 네트워크(Plain Network)가 $a^{[l+2]} = g(z^{[l+2]})$를 학습한다면, Residual Block은 입력 $a^{[l]}$을 두 레이어 뒤로 바로 더해줍니다:

$$a^{[l+2]} = g(z^{[l+2]} + a^{[l]})$$

### **ResNet이 잘 작동하는 이유**

- 두 레이어의 가중치 $W^{[l+2]}, b^{[l+2]}$가 0에 가깝게 대폭 감소하더라도, $g(a^{[l]}) = a^{[l]}$ 형태가 되어 **최소한 항등 함수(Identity Function)를 쉽게 학습**해냅니다.
    
- 즉, 레이어를 아무리 깊게 쌓아도 이전 레이어만큼의 성능은 최소한 보장하므로 깊은 모델의 학습이 매우 쉬워집니다.
    

## 3. $1 \times 1$ 합성곱과 Inception 네트워크

### **1) $1 \times 1$ Convolution (Network in Network)**

채널 수가 많은 특징 맵에서 가로/세로 해상도는 유지하면서 **채널 수만 줄이거나 늘리는 차원 축소/확장(Dimension Reduction)** 역할을 수행합니다.

### **2) Inception 모듈 (GoogleNet, 2014)**

"필터 크기($1 \times 1, 3 \times 3, 5 \times 5$)나 Pooling 중 무엇을 쓸지 고민하지 말고, **모두 다 적용한 뒤 채널 방향으로 합치자(Concatenate)**"는 아이디어입니다.

- **Bottleneck Layer:** $5 \times 5$ CONV를 직접 적용하면 연산량이 폭발하므로, 중간에 **$1 \times 1$ CONV를 먼저 거쳐 채널 수를 줄인 후** $5 \times 5$ 연산을 수행함으로써 연산 비용을 1/10 수준으로 절감합니다.
    

## 4. 경량화 및 효율적 모델 (MobileNet & EfficientNet)

모바일/임베디드 기기와 같이 연산 자원이 제한된 환경을 위해 설계된 아키텍처입니다.

### **1) MobileNet (v1, v2)**

- **Depthwise Separable Convolution:** 연산량을 감축하기 위해 표준 합성곱을 2단계로 분리합니다.
    
    1. **Depthwise Convolution:** 채널별로 각각 $f \times f$ 필터 적용.
        
    2. **Pointwise Convolution:** $1 \times 1$ CONV로 채널들을 하나로 연산.
        
    
    - 일반 CONV 대비 연산량을 약 $\frac{1}{N} + \frac{1}{f^2}$ (약 10~15%) 수준으로 획기적으로 단축합니다.
        
- **MobileNet v2 (Inverted Residuals & Linear Bottlenecks):**
    
    특징 맵의 채널을 확장시켰다가 다시 압축하는 **Inverted Residual Block**과 릴루(ReLU)로 인한 정보 손실을 막기 위한 **Linear Bottleneck**을 적용했습니다.
    

### **2) EfficientNet**

컴퓨터 비전 모델의 성능을 올리기 위한 3가지 스케일링 요소인 Width(채널 수), Depth(레이어 깊이), Resolution(입력 해상도)를 복합적으로 동시에 균형 있게 조절하는 **Compound Scaling** 기법을 적용하여 적은 파라미터로 최적의 성능을 냅니다.

## 5. 실전 구현 테크닉 (Practical Advice for Computer Vision)

### **1) 전이 학습 (Transfer Learning)**

- 이미 대규모 데이터셋(ImageNet 등)으로 학습된 오픈소스 가중치를 가져와 다운로드합니다.
    
- 데이터가 적을 때는 기존 가중치를 Freeze(동결)하고 마지막 Softmax 레이어만 재학습시킵니다.
    
- 데이터가 충분하다면 앞단 일부 레이어만 동결하고 나머지 레이어 전체를 Fine-tuning(미세 조정)합니다.
    

### **2) 데이터 증강 (Data Augmentation)**

데이터가 부족한 컴퓨터 비전 과제에서 오버피팅을 방지하기 위한 필수 기술입니다.

- **기본 기법:** Mirroring(좌우 반전), Random Cropping(랜덤 자르기), Rotation(회전), Shear(전단 변형)
    
- **색상 변형 (Color Jittering):** RGB 채널의 밝기, 대비, 채도 변경 (PCA Color Augmentation 등)
    

### **3) 캐글/경쟁 대회 팁 (State of Computer Vision)**

- **Ensembling:** 서로 다른 하이퍼파라미터로 학습된 3~5개 모델의 예측값을 평균 내어 성능을 1~2% 향상시킵니다.
    
- **Multi-crop at test time (10-crop):** 테스트 이미지를 중앙/모서리 5개 위치에서 크롭하고 반전시켜 총 10장 이미지의 예측 결과를 평균 냅니다.
    
    _(단, 실무 배포 시에는 연산 속도 저하로 인해 잘 쓰이지 않습니다.)_
    

