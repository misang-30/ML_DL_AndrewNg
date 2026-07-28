Matplotlib은 딥러닝에서 **데이터의 상태를 눈으로 확인(시각화)하고 모델의 학습 상황을 모니터링하는 도구**입니다.

화려한 차트를 만드는 고급 그래픽 스킬까지는 필요 없으며, "원하는 데이터를 바로 그래프로 띄워서 문제점을 찾아낼 수 있는 정도"만 아시면 충분합니다. 

## 1. 딥러닝 실무에 꼭 필요한 Matplotlib 핵심 스킬 3가지

### 1). 이미지 데이터 확인하기 (`imshow`)

이미지 데이터를 다룰 때 텐서(Tensor)가 제대로 전처리되었는지 눈으로 확인할 때 가장 많이 쓰입니다.


```Python

import matplotlib.pyplot as plt
import torch

# (1). 이미지 텐서 생성 (예: PyTorch 형태 C, H, W -> 3, 224, 224)
img_tensor = torch.randn(3, 224, 224)

# (2). PyTorch 차원(C, H, W)을 Matplotlib 형태(H, W, C)로 변경 (permute 활용)
img_to_show = img_tensor.permute(1, 2, 0).numpy()

# (3). 화면에 출력
plt.imshow(img_to_show)
plt.axis('off')  # 눈금 제거
plt.show()
```

### 2). 학습 손실(Loss) 및 정확도(Accuracy) 곡선 그리기 (`plot`)

모델이 오버피팅(과적합)되고 있는지, 학습이 잘되고 있는지 확인할 때 필수적입니다.



```Python
import matplotlib.pyplot as plt

train_losses = [0.8, 0.6, 0.4, 0.3, 0.2]
val_losses = [0.85, 0.65, 0.5, 0.45, 0.48]  # Validation Loss

# (1). 선 그래프 그리기
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Val Loss', color='red', linestyle='--')

# (2). 라벨 및 범례 추가
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()  # 범례 표시
plt.grid(True)  # 격자 표시

plt.show()
```

### 3). 여러 개 이미지를 한 번에 띄우기 (`plt.subplots`)

배치(Batch) 단위의 여러 이미지 데이터를 한 화면에서 동시에 점검할 때 사용합니다.



```Python
import matplotlib.pyplot as plt
import numpy as np

# 2행 4열(총 8개) 서브플롯 생성
fig, axes = plt.subplots(2, 4, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    # 임의의 데이터 생성
    dummy_img = np.random.rand(28, 28)
    
    ax.imshow(dummy_img, cmap='gray')
    ax.set_title(f'Sample {i+1}')
    ax.axis('off')

plt.tight_layout()  # 여백 자동 조절
plt.show()
```

## 2. 결론: 딱 이 정도면 끝입니다

|**구분**|**필요 스킬**|**용도**|
|---|---|---|
|**필수**|`plt.plot()`|Loss, Accuracy 등의 학습 추이 모니터링|
|**필수**|`plt.imshow()`|이미지 데이터 시각화 및 전처리 결과 확인|
|**필수**|`plt.subplots()`|여러 데이터/결과물을 한 번에 비교 분석|
|**선택**|`plt.hist()`, `plt.scatter()`|데이터 분포/이상치 분석 (필요할 때 검색)|

> 디자인을 예쁘게 꾸미는 옵션(폰트 변경, 색상 조정 등)을 외우실 필요는 전혀 없습니다. **"데이터나 손실값을 넣어서 창에 띄울 수 있는 정도"**만 갖추시고, 복잡한 시각화가 필요한 순간이 오면 그때그때 검색해 쓰시면 됩니다.