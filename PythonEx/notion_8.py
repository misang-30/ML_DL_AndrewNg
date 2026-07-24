# ---------------<PL08-리스트 사용법 >---------------

# 1). 리스트 사용 1 ---------------------------------------
# Create list with a string and a nested list of two strings.
al = ['Weird Al', ['Like a Surgeon', 'Perform this Way']]

# Check length of al.
print(len(al))
# 2

# Access elements by index
print(al[0])
# Weird Al
print(al[1])
# ['Like a Surgeon', 'Perform this Way']

# Cycle through the elements of list al.
for item in al:
    print(item)
# Weird Al
# ['Like a Surgeon', 'Perform this Way']


# 2). 리스트 사용 2 ---------------------------------------
# Create a list of three nested lists, each of which contains
# a string and one or two integers.
produce = [['carrots', 56],
           ['celery', 178, 198],
           ['bananas', 59]]

# 리스트 전체 출력
print(produce)
# 결과: [['carrots', 56], ['celery', 178, 198], ['bananas', 59]]

# 인덱스와 함께 리스트 요소 출력
for i in range(len(produce)):
    print(i, produce[i])

# 위 반복문의 출력 결과:
# 0 ['carrots', 56]
# 1 ['celery', 178, 198]
# 2 ['bananas', 59]


# 3). 리스트 사용 3 ---------------------------------------
# Create individual artists as lists consisting of a name and a list of songs.
al = ["Weird Al", ["Like a Surgeon", "Perform this Way"]]
gaga = ["Lady Gaga", ["Bad Romance", "Born this Way"]]
madonna = ["Madonna", ["Like a Virgin", "Papa Don't Preach"]]

# Collect individual artists together in one list of artists.
artists = [al, gaga, madonna]

# 아티스트 리스트 전체 출력
print(artists)
# 출력 결과: [['Weird Al', ['Like a Surgeon', 'Perform this Way']], ['Lady Gaga', ['Bad Romance', 'Born this Way']], ['Madonna', ['Like a Virgin', "Papa Don't Preach"]]]

# 인덱스를 활용한 반복문 출력
for i in range(len(artists)):
    print(i, artists[i])

# 위 반복문의 출력 결과:
# 0 ['Weird Al', ['Like a Surgeon', 'Perform this Way']]
# 1 ['Lady Gaga', ['Bad Romance', 'Born this Way']]
# 2 ['Madonna', ['Like a Virgin', "Papa Don't Preach"]]







# 4). 리스트 사용 4 ---------------------------------------
# Create individual artists as lists consisting of a name and a list of songs.
al = ["Weird Al", ["Like a Surgeon", "Perform this Way"]]
gaga = ["Lady Gaga", ["Bad Romance", "Born this Way"]]
madonna = ["Madonna", ["Like a Virgin", "Papa Don't Preach"]]

# Collect individual artists together in one list of artists.
artists = [al, gaga, madonna]

# 아티스트 리스트 전체 출력
print(artists)
# 출력 결과: [['Weird Al', ['Like a Surgeon', 'Perform this Way']], ['Lady Gaga', ['Bad Romance', 'Born this Way']], ['Madonna', ['Like a Virgin', "Papa Don't Preach"]]]

# 인덱스를 활용한 반복문 출력
for i in range(len(artists)):
    print(i, artists[i])

# 위 반복문의 출력 결과:
# 0 ['Weird Al', ['Like a Surgeon', 'Perform this Way']]
# 1 ['Lady Gaga', ['Bad Romance', 'Born this Way']]
# 2 ['Madonna', ['Like a Virgin', "Papa Don't Preach"]]



# 5). 리스트 사용 5 ---------------------------------------


toyota = ["Toyota", ["Prius", "4Runner", "Sienna", "Camry"]]

# 첫 번째 요소 접근
print(toyota[0]) 
# 결과: 'Toyota'

# 두 번째 요소(하위 리스트) 접근
print(toyota[1]) 
# 결과: ['Prius', '4Runner', 'Sienna', 'Camry']

# 하위 리스트의 특정 요소 접근 (이중 인덱싱)
print(toyota[1][0]) # 결과: 'Prius'
print(toyota[1][3]) # 결과: 'Camry'

# 리스트 길이 확인
print(len(toyota))    # 외부 리스트의 길이. 결과: 2
print(len(toyota[1])) # 내부(하위) 리스트의 길이. 결과: 4

# 마지막 요소 접근의 논리적 방법
print(toyota[1][len(toyota[1]) - 1]) 
# 결과: 'Camry'

for model in toyota[1]:
    print(model)
# 결과:
# Prius
# 4Runner
# Sienna
# Camry



# 6). 리스트 사용 6 ---------------------------------------
def show_brand(brand):
    print("Make:", brand[0])
    print(" Model:")
    for i in range(len(brand[1])):
        print("    ", i + 1, brand[1][i])

toyota = ["Toyota", ["Prius", "4Runner", "Sienna", "Camry"]]
ford = ["Ford", ["Focus", "Taurus", "Mustang", "Fusion", "Fiesta"]]

show_brand(toyota)
# 결과:
# Make: Toyota
#  Model:
#      1 Prius
#      2 4Runner
#      3 Sienna
#      4 Camry

show_brand(ford)
# 결과:
# Make: Ford
#  Model:
#      1 Focus
#      2 Taurus
#      3 Mustang
#      4 Fusion
#      5 Fiesta


def show_nested_lists(xlist):
    for i in range(len(xlist)):
        for j in range(len(xlist[i])):
            print("xlist[", i, "][", j, "] = ", xlist[i][j], sep="")
        print()

produce = [['carrots', 56], ['celery', 178, 198], ['bananas', 59]]
show_nested_lists(produce)

# 실행 결과:
# xlist[0][0] = carrots
# xlist[0][1] = 56
#
# xlist[1][0] = celery
# xlist[1][1] = 178
# xlist[1][2] = 198
#
# xlist[2][0] = bananas
# xlist[2][1] = 59


# 7). 리스트의 동시 할당 ---------------------------------------

def show_nested_lists(xlist):
    for i in range(len(xlist)):
        for j in range(len(xlist[i])):
            print("xlist[", i, "][", j, "] = ", xlist[i][j], sep="")
        print()

produce = [['carrots', 56], ['celery', 178, 198], ['bananas', 59]]
show_nested_lists(produce)

# 결과:
# xlist[0][0] = carrots
# xlist[0][1] = 56
#
# xlist[1][0] = celery
# xlist[1][1] = 178
# xlist[1][2] = 198
#
# xlist[2][0] = bananas
# xlist[2][1] = 59

def show_artists(artists):
    for artist in artists:
        name, songs = artist # 이름과 노래 리스트로 언패킹
        print(name)
        for song in songs:
            print("    ", song)

performers = [
    ["Weird Al", ["Like a Surgeon", "Perform this Way"]],
    ["Lady Gaga", ["Bad Romance", "Born this Way"]],
    ["Madonna", ["Like a Virgin", "Papa Don't Preach"]]
]

show_artists(performers)

# 결과:
# Weird Al
#      Like a Surgeon
#      Perform This Way
# Lady Gaga
#      Bad Romance
#      Born This Way
# Madonna
#      Like a Virgin
#      Papa Don't Preach

shoes = [["Manolo Blahnik", 120], ["Bontoni", 96],
         ["Maud Frizon", 210], ["Tinker Hatfield", 54],
         ["Lauren Jones", 88], ["Beatrix Ong", 150]]

for designer, price in shoes:
    print(designer, ": $", price, sep="")

# 실행 결과:
# Manolo Blahnik: $120
# Bontoni: $96
# Maud Frizon: $210
# Tinker Hatfield: $54
# Lauren Jones: $88
# Beatrix Ong: $150

# 튜플 리스트 생성 및 출력
flist = [(21, 'apples'), (17, 'bananas'), (39, 'oranges')]
for count, fruit in flist:
    print(count, fruit)

# 결과:
# 21 apples
# 17 bananas
# 39 oranges

# 튜플의 튜플 생성 및 출력
ftuple = ((21, 'apples'), (17, 'bananas'), (39, 'oranges'))
for count, fruit in ftuple:
    print(count, fruit)

# 결과:
# 21 apples
# 17 bananas
# 39 oranges



# 8). 레퍼런스, 리스트의 가변성 ---------------------------------------

"""

• 파이썬에서 리스트는 수백, 수천, 심지어 수백만 개의 원소를 가질 수 있으며,
 리스트 안의 데이터는 컴퓨터 메모리에서 상당한 공간을 차지할 수 있다.
• 따라서 리스트를 새로운 변수에 할당할 때, 기본 동작은
 이 새로운 변수를 위해 기본 데이터의 독립적인 복사본을 만드는 것이 아니다.
• 대신, 그 할당은 새로운 변수가 같은 기본 데이터를 가리키는 참조(reference)
 (또는 별명, alias) 가 되도록 만든다.
 즉, 원래 리스트가 저장된 같은 메모리 위치를 가리키게 된다.

"""
"""
Namespace에는 1).Names , 2).Objects  2가지가 있다.
Names는 변수이고, Object는 실체가 있는 메모리라고 볼 수 있다.
**리스트는 레퍼런스 형태로 바인딩하는 것.**
따라서, 리스트 할당하는 경우, 바인딩할 대상을 알려주는 것일 뿐이라서 아래 코드가 가능한 것이다.



"""


xlist = [11] # Create xlist with a single element.
ylist = xlist # Make ylist an alias for xlist.
print(xlist, ylist)
# 출력 결과: [11] [11] # Show xlist and ylist.

xlist[0] = 22 # Change value of the element in xlist.
print(xlist, ylist)
# 출력 결과: [22] [22] # Change is visible in both xlist and ylist.

ylist[0] = 33 # Change value of the element in ylist.
print(xlist, ylist)
# 출력 결과: [33] [33] # Change is visible in both xlist and ylist.

# Use built-in is operator to show xlist and ylist are the same.
xlist is ylist
# 출력 결과: True

xlist = [11] # Create xlist with a single element.
ylist = xlist # Make ylist an alias for xlist.
print(xlist, ylist)
# 출력 결과: [11] [11] # Show xlist and ylist.

xlist is ylist # See that xlist and ylist are aliases.
# 출력 결과: True

xlist = [22] # Change xlist to point to a new list.
print(xlist, ylist)
# 출력 결과: [22] [11] # See that xlist changes, but not ylist.

xlist is ylist # See that xlist and ylist are no longer the same.
# 출력 결과: False

def doubler(xlist):
    for i in range(len(xlist)):
        xlist[i] = 2 * xlist[i]

zlist = [1, 2, 3] # Create list of integers.
print(zlist)
# 출력 결과: [1, 2, 3] # Check contents.

doubler(zlist) # Call doubler().
print(zlist)
# 출력 결과: [2, 4, 6] # See that contents have doubled.


# 9). string as Iterable or Squences ---------------------------------------

hi = "Hello World!"
hi[0] # First character.
# 출력 결과: 'H'

hi[6] # Seventh character.
# 출력 결과: 'W'

len(hi) # Length of string.
# 출력 결과: 12

hi[len(hi) - 1] # Last character.
# 출력 결과: '!'

print(hi[0], hi[1], hi[5], hi[7], hi[9], hi[10], hi[11], sep = "")
# 출력 결과: He old!

yo = "Hey!"
for i in range(len(yo)):
    print(i, yo[i])

# 출력 결과:
# 0 H
# 1 e
# 2 y
# 3 !

# 10). Negative Indices ---------------------------------------

#문자,H,e,l,l,o,!
#Positive indices (양수 인덱스),0,1,2,3,4,5
#Negative indices (음수 인덱스),-6,-5,-4,-3,-2,-1



# 11). Slicing ---------------------------------------

# 1. 문자열 슬라이싱 기초 (s[start : stop])
s = "Washington State"

print(s[1 : 4])        # 출력: 'ash' (인덱스 1부터 3까지)
print(s[ : 4])         # 출력: 'Wash' (처음부터 인덱스 3까지)
print(s[4 : 6])        # 출력: 'in' (인덱스 4부터 5까지)
print(s[4 : ])         # 출력: 'ington State' (인덱스 4부터 끝까지)
print(s[-3 : ])        # 출력: 'ate' (뒤에서 3번째부터 끝까지)

# 2. 음수 및 혼합 인덱스 활용
print(s[4 : -3])       # 출력: 'ington St' (s[4:13]과 동일)
print(s[-10 : 10])     # 출력: 'gton' (s[6:10]과 동일)
print(s[-9 : -2])      # 출력: 'ton Sta' (음수 인덱스 구간)
print(s[7 : 14])       # 출력: 'ton Sta' (위와 동일한 양수 표현)
print(s[ : ])          # 출력: 'Washington State' (전체 복사)

# 3. 증감값(Increment/Step) 활용 (s[start : stop : step])
print(s[ : : 2])       # 출력: 'Wsigo tt' (2칸 간격으로)
print(s[5 : : 2])      # 출력: 'ntnSae' (5번 인덱스부터 2칸 간격)
print(s[5 : -1 : 2])   # 출력: 'ntnSa' (5번부터 마지막 글자 제외하고 2칸 간격)
print(s[ : : -1])      # 출력: 'etatS notgnihsaW' (문자열 뒤집기)
print(s[10 : 1 : -1])  # 출력: 'notgnihs' (10번부터 2번까지 역순)

# 4. 범위를 벗어난 슬라이싱 (에러 발생 안 함)
xlist = ['a', 'b', 'c']
print(xlist[1 : 100])      # 출력: ['b', 'c']
print(xlist[-100 : 1])     # 출력: ['a']
print(xlist[-100 : 100])   # 출력: ['a', 'b', 'c']

# 참고: 개별 인덱싱은 범위를 벗어나면 에러 발생
# print(xlist[100])  # IndexError 발생

# 5. 리스트 복사 vs 별칭(Alias)
xlist = [1, 2, 3]
ylist = xlist[:]    # 독립적인 복사본 생성 (Slicing 이용)
zlist = xlist       # 별칭 생성 (같은 메모리 주소 참조)

print(xlist, ylist, zlist) # 출력: [1, 2, 3] [1, 2, 3] [1, 2, 3]

xlist[1] = 200      # 원본 수정
print(xlist, ylist, zlist) 
# 출력: [1, 200, 3] [1, 2, 3] [1, 200, 3] (ylist는 안 변함)

zlist[0] = 100      # 별칭 수정
print(xlist, ylist, zlist) 
# 출력: [100, 200, 3] [1, 2, 3] [100, 200, 3] (xlist도 같이 변함)

print(xlist is ylist) # 출력: False (서로 다른 객체)



# 12). list Comprehension ---------------------------------------
xlist = ['a', 'b', 'c']

# 슬라이싱: 중단 값이 끝을 넘어가는 경우
xlist[1 : 100]
# 출력: ['b', 'c']

# 슬라이싱: 시작 값이 처음보다 앞서는 경우
xlist[-100 : 1]
# 출력: ['a']

# 슬라이싱: 시작과 중단 모두 범위를 벗어난 경우
xlist[-100 : 100]
# 출력: ['a', 'b', 'c']

# 인덱싱: 개별 요소가 범위를 벗어난 경우 (음수)
xlist[-100]
# 출력: IndexError: list index out of range

# 인덱싱: 개별 요소가 범위를 벗어난 경우 (양수)
xlist[100]
# 출력: IndexError: list index out of range

mylist = [1, 2, 3]

# 리스트 컴프리헨션으로 새로운 리스트 생성
[2 * x for x in mylist]
# 출력: [2, 4, 6]

def lcdoubler(xlist):
    return [2 * x for x in xlist]

doubledlist = lcdoubler(mylist)

doubledlist
# 출력: [2, 4, 6]

mylist
# 출력: [1, 2, 3] (원본 리스트는 유지됨)

def newdoubler(xlist):
    doublelist = []
    for x in xlist:
        doublelist.append(2 * x)
    return doublelist

mylist = [1, 2, 3]

newdoubler(mylist)
# 출력: [2, 4, 6]

doubledlist = newdoubler(mylist)

doubledlist
# 출력: [2, 4, 6]

mylist
# 출력: [1, 2, 3]

mylist = [1, 2, 3]

# 인덱스를 사용하여 2배 만들기
[2 * mylist[i] for i in range(len(mylist))]
# 출력: [2, 4, 6]

def scaler(xlist):
    # 인덱스 번호에 1을 더한 값을 곱함
    return [(i + 1) * xlist[i] for i in range(len(xlist))]

scaler(mylist)
# 출력: [1, 4, 9] (1*1, 2*2, 3*3)

scaler(['a', 'b', 'c', 'd'])
# 출력: ['a', 'bb', 'ccc', 'dddd'] (문자열 곱셈 적용)