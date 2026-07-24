## < 목차 >

1. NumPy 기초 및 배열 생성
    - NumPy의 존재 이유 및 list와의 차이
    - 1D, 2D 배열 생성 (`np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`)
    - 배열의 핵심 속성 확인 (`shape`, `dtype`, `ndim`)
    - 목표: 파이썬 리스트를 NumPy 배열로 바꾸고 원하는 모양의 기본 배열을 자유롭게 생성하기
        
2. 차원 변경 및 인덱싱 / 슬라이싱
    - 차원 변경 (`reshape`, `flatten`, `transpose`)
    - 1D, 2D 배열 인덱싱 및 슬라이싱
    - 조건부 필터링 (Boolean Indexing 및 `np.where`)
    - 목표: 원하는 위치의 데이터만 추출하고 배열의 차원을 자유자재로 조작하기
        
3. 기본 연산 및 브로드캐스팅 (Broadcasting
    - 요소별 연산 (Element-wise Operations)
    - 브로드캐스팅 규칙 (다른 모양의 배열 간 연산)
    - 축(Axis) 기반 통계 연산 (`sum`, `mean`, `max`, `std` + `axis` 개념)
    - 목표: 반복문 없이 전체 데이터의 연산 및 통계치 추출하기
        
4. 배열 결합 및 분할, 행렬 연산
    - 배열 붙이기 및 쪼개기 (`np.concatenate`, `np.vstack`, `np.hstack`)
    - 행렬 곱 연산 (`np.dot`, `@` 연산자)
    - 내장 함수 활용 (`np.unique`, `np.sort`, `np.argsort`)
    - 목표: 여러 배열을 결합/분할하고 선형대수 기본 연산 수행하기

5. 난수 생성 및 데이터 셔플링 (Random Sampling)
    - 시드 고정 (`np.random.seed` / `default_rng`)
    - 난수 배열 생성 (`rand`, `randn`, `randint`)
    - 데이터 섞기 및 복원/비복원 추출 (`shuffle`, `choice`)
    - 목표: 실험 재현성을 확보하고 학습 데이터를 무작위로 샘플링하기
        
6. 구조화된 조건 검색 및 필터링
    - 다중 조건 검사 (`np.logical_and`, `np.logical_or`)
    - 조건 만족 여부 확인 (`np.any`, `np.all`)
    - 값 제한 및 이상치 처리 (`np.clip`)
    - 목표: 복잡한 조건문 없이 데이터 품질을 정제하고 이상치 판별하기
        
7. 차원 조작 및 확장 고급
    - 차원 추가 (`np.newaxis`, `np.expand_dims`)
    - 불필요한 차원 제거 (`np.squeeze`)
    - 배열 재배치 (`np.moveaxis`, `np.swapaxes`)
    - 목표: 파이토치 모델 입력 형태에 맞게 텐서 차원 전처리를 완벽하게 맞추기
        
8. 파일 입출력 및 메모리 효율화
    - NumPy 전용 바이너리 파일 저장/로드 (`np.save`, `np.load`)
    - 텍스트/CSV 파일 읽기 (`np.loadtxt`, `np.genfromtxt`)
    - 목표: 전처리된 대용량 데이터를 빠르게 저장하고 불러오기

---
---
## < NumPy 핵심 개념 정리 >

### 1. NumPy 기초 및 배열 생성

NumPy는 연속된 메모리 공간에 데이터를 저장하므로 파이썬 기본 리스트보다 연산 속도가 빠르고 메모리 효율이 높습니다. 파이토치 텐서(Tensor)의 기반이 되는 데이터 구조입니다.


``` Python
import numpy as np

# 1. 일반 리스트를 NumPy 배열로 변환
arr = np.array([1, 2, 3, 4])
print(arr)

# 2. 자주 쓰는 기본 배열 생성
zeros_arr = np.zeros((2, 3))       # 0으로 채워진 2x3 배열
ones_arr = np.ones((2, 3))         # 1로 채워진 2x3 배열
range_arr = np.arange(0, 10, 2)    # 0부터 10 미만까지 2씩 증가 [0, 2, 4, 6, 8]
linspace_arr = np.linspace(0, 1, 5)# 0부터 1까지 균등하게 5개로 나눔

# 3. 배열 속성 확인 (가장 중요)
data = np.array([[1, 2, 3], [4, 5, 6]])
print(data.shape)  # 차원 모양: (2, 3)
print(data.dtype)  # 데이터 타입: int64 또는 int32
print(data.ndim)   # 차원 수: 2
```

> **핵심 포인트:**
> 
> - NumPy 배열은 단일 데이터 타입만 저장 가능합니다. (예: 정수와 실수가 섞이면 모두 실수로 자동 변환)
>     
> - `shape`는 데이터를 처리할 때 차원 오류를 막기 위해 항상 확인해야 하는 속성입니다.
>     

### 2. 차원 변경 및 인덱싱 / 슬라이싱

데이터의 모양을 변형하거나 필요한 부분만 선택하여 추출하는 방법입니다.


``` Python
import numpy as np

arr = np.arange(12) # [0, 1, 2, ..., 11]

# 1. 차원 변경 (Reshape)
# -1은 남은 요소 개수에 맞춰 자동으로 차원 크기를 계산합니다.
matrix = arr.reshape(3, 4)  # 3행 4열로 변환
flat = matrix.reshape(-1)   # 다시 1차원으로 펴기

# 2. 인덱싱 및 슬라이싱
# matrix[행_슬라이싱, 열_슬라이싱]
print(matrix[0, 1])     # 0행 1열의 값
print(matrix[:2, 1:3])  # 0~1행, 1~2열 부분 행렬 추출

# 3. 불리언 인덱싱 (Boolean Indexing) - 조건에 맞는 값만 추출
scores = np.array([80, 95, 55, 70, 100])
mask = scores >= 80
print(scores[mask]) # [80, 95, 100]

# 4. np.where - 조건에 따른 값 변경
# 조건, 참일 때 값, 거짓일 때 값
result = np.where(scores >= 80, "Pass", "Fail")
```

> **핵심 포인트:**
> 
> - `reshape`할 때 전체 요소 개수가 유지되어야 합니다. (12개 데이터를 3x4로 바꿀 수는 있지만 3x5로는 바꿀 수 없음)
>     
> - 불리언 인덱싱은 파이썬의 `for`문과 `if`문을 대체하여 코드를 간결하게 만듭니다.
>     

### 3. 기본 연산 및 브로드캐스팅 (Broadcasting)

NumPy는 반복문 없이 배열 전체에 대해 한 번에 연산을 수행합니다.

``` Python
import numpy as np

# 1. 요소별 연산 (Element-wise)
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
print(a + b) # [11, 22, 33]
print(a * 2) # [2, 4, 6]

# 2. 브로드캐스팅 (Broadcasting)
# 모양이 다른 배열 간의 연산을 자동으로 확장하여 수행
matrix = np.array([[1, 2, 3], [4, 5, 6]]) # (2, 3)
row_vec = np.array([10, 20, 30])           # (3,)
print(matrix + row_vec)                    # row_vec이 각 행에 더해짐

# 3. 축(Axis) 기반 통계 연산
# axis=0: 행 방향(열 기준 아래로 연산)
# axis=1: 열 방향(행 기준 옆으로 연산)
arr2d = np.array([[1, 2], [3, 4]])
print(arr2d.sum(axis=0)) # [4, 6] (각 열의 합)
print(arr2d.mean(axis=1)) # [1.5, 3.5] (각 행의 평균)
```

> **핵심 포인트:**
> 
> - `axis=0`은 위에서 아래로(컬럼별), `axis=1`은 좌에서 우로(로우별) 연산이 실행됩니다.
>     
> - 브로드캐스팅은 두 배열의 차원 크기가 같거나, 둘 중 하나가 1일 때 적용됩니다.
>     

### 4. 배열 결합 및 분할, 행렬 연산

여러 데이터를 합치거나 분할하고 선형대수의 기본 연산을 처리하는 방법입니다.


``` Python
import numpy as np

# 1. 배열 결합 (Concatenate, vstack, hstack)
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

v_stacked = np.vstack((a, b)) # 수직으로 쌓기 (4, 2)
h_stacked = np.hstack((a, b)) # 수평으로 쌓기 (2, 4)

# 2. 행렬 곱 연산 (Dot Product)
# 단순 곱(a * b)은 요소별 곱이며, 행렬 곱은 @ 연산자 또는 np.dot 사용
m1 = np.array([[1, 2], [3, 4]]) # (2, 2)
m2 = np.array([[5, 6], [7, 8]]) # (2, 2)
dot_result = m1 @ m2            # 행렬 곱 계산

# 3. 유용한 내장 함수
arr = np.array([3, 1, 2, 2, 5, 1])
print(np.unique(arr))  # 중복 제거: [1, 2, 3, 5]
print(np.sort(arr))    # 정렬: [1, 1, 2, 2, 3, 5]
print(np.argsort(arr)) # 정렬되었을 때의 원본 인덱스 위치 반환
```

> **핵심 포인트:**
> 
> - 행렬 곱 연산(`@`)을 수행하려면 앞 행렬의 열 개수와 뒤 행렬의 행 개수가 일치해야 합니다. (예: `(A, B) @ (B, C) -> (A, C)`)
>     
> - `argsort`는 정렬된 순서의 위치(인덱스)를 반환하므로 데이터 정렬 및 상위 N개 추출 시 자주 사용됩니다.



---
---
## < Numpy 심화 개념 >
### 5. 난수 생성 및 데이터 셔플링

데이터셋을 트레이닝/테스트용으로 나누거나 가중치를 초기화할 때 사용합니다.

``` Python
import numpy as np

# 1. 시드(Seed) 고정: 언제 실행해도 똑같은 난수가 나오도록 설정 (재현성 확보)
rng = np.random.default_rng(seed=42)

# 2. 다양한 난수 생성
uniform_arr = rng.random((2, 3))       # 0~1 사이 균일 분포 (2x3)
normal_arr = rng.standard_normal((2, 3)) # 평균 0, 표준편차 1의 정규분포
int_arr = rng.integers(0, 10, size=5)   # 0 이상 10 미만의 정수 5개

# 3. 데이터 섞기 (In-place로 원본이 변경됨)
data = np.array([10, 20, 30, 40, 50])
rng.shuffle(data)

# 4. 샘플링 (Choice)
# data에서 3개를 비복원(replace=False) 추출
sampled = rng.choice(data, size=3, replace=False)
```

> **핵심 포인트:**
> 
> - 최근 NumPy 버전에선 `np.random.seed()` 대신 `np.random.default_rng()` 사용을 권장합니다.
>     
> - `replace=False` 옵션을 주면 중복 없이 뽑을 수 있어 데이터 분할 시 유용합니다.
>     

### 6. 구조화된 조건 검색 및 필터링

이상치(Outlier)를 제거하거나 복잡한 조건으로 데이터를 정제할 때 사용합니다.

```Python
import numpy as np

data = np.array([15, 88, 42, 105, 73, -5])

# 1. 다중 조건 검사 (&: AND, |: OR, ~: NOT)
# 각 조건은 반드시 괄호()로 묶어야 합니다.
mask = (data >= 10) & (data <= 90)
filtered_data = data[mask] # [15, 88, 42, 73]

# 2. np.any, np.all
print(np.any(data < 0))   # 음수가 하나라도 있는가? -> True
print(np.all(data > 0))   # 모든 요소가 양수인가? -> False

# 3. np.clip (최댓값, 최솟값 제한)
# -5는 0으로, 105는 100으로 강제 조정
clipped_data = np.clip(data, a_min=0, a_max=100)
# 결과: [15, 88, 42, 100, 73, 0]
```

> **핵심 포인트:**
> 
> - 비트 연산자(`&`, `|`)를 쓸 때는 연산자 우선순위 때문에 조건식을 `(data > 0)` 형태로 괄호 처리해야 합니다.
>     
> - `np.clip`은 딥러닝에서 그래디언트 폭주를 막거나 이미지 픽셀값(0~255)을 제한할 때 자주 쓰입니다.
>     

### 7. 차원 조작 및 확장 고급

파이토치나 캐라스 같은 딥러닝 프레임워크와 연동할 때 차원(Batch, Channel 등)을 맞춰주기 위해 필수적입니다.

```Python
import numpy as np

arr = np.array([1, 2, 3, 4]) # shape: (4,)

# 1. 차원 추가 (np.newaxis 또는 np.expand_dims)
row_vec = arr[np.newaxis, :]  # shape: (1, 4) - 행 방향 차원 추가
col_vec = arr[:, np.newaxis]  # shape: (4, 1) - 열 방향 차원 추가

expanded = np.expand_dims(arr, axis=0) # shape: (1, 4)

# 2. 크기가 1인 불필요한 차원 제거 (np.squeeze)
squeezed = np.squeeze(expanded) # shape: (4,)

# 3. 차원 위치 변경 (Transpose / Swapaxes)
img = np.zeros((224, 224, 3)) # (Height, Width, Channel) - OpenCV 형태
# 파이토치는 (Channel, Height, Width) 순서를 요구하므로 축 변경 필요
img_pt = np.moveaxis(img, -1, 0) # shape: (3, 224, 224)
```

> **핵심 포인트:**
> 
> - 이미지를 파이토치로 넘길 때 `(H, W, C)` 구조를 `(C, H, W)`로 바꾸는 작업을 자주 만납니다. 이때 `np.moveaxis`나 `transpose`를 사용합니다.
>     

### 8. 파일 입출력

전처리된 배열을 텍스트가 아닌 압축된 바이너리 형태(`.npy`)로 저장하면 읽기/쓰기 속도가 훨씬 빠릅니다.

```Python
import numpy as np

data_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 1. 단일 배열 저장 및 불러오기 (.npy)
np.save("my_array.npy", data_matrix)
loaded_matrix = np.load("my_array.npy")

# 2. 여러 배열을 하나의 파일로 저장 (.npz)
a = np.arange(5)
b = np.arange(10)
np.savez("multi_array.npz", array_a=a, array_b=b)

# 불러오기
loaded_npz = np.load("multi_array.npz")
print(loaded_npz["array_a"])
print(loaded_npz["array_b"])
```

> **핵심 포인트:**
> 
> - `.npy` 및 `.npz` 파일 형식은 파이썬 객체의 타입을 그대로 유지하면서 매우 빠른 속도로 데이터를 로드할 수 있어 딥러닝 전처리 데이터 저장용으로 쓰입니다.
>