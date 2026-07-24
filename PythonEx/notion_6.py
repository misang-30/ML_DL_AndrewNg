# ---------------<PL06-Introduction to Objects>---------------
# >>>0. 객체 지향 
"""
OOP는 프로그램을 구성하는 방식을 사람이 사고하는 방식과 더 유사하게 만들어 줍니다.

즉, 비객체 지향(non-OOP) 방식보다 사람의 사고 방식과 더 닮았습니다.

따라서 많은 경우에 OOP 접근 방식은 문제를 해결하는 가장 명확하고 단순한 경로를 제공합니다.

그러나 상상할 수 있듯이, 인간의 사고 방식을 반영하려고 할수록 미묘한 점과 
여러 층의 복잡성이 존재할 수 있습니다.

"""

"""
데이터를 생각하고 함수와 연결해 보면, 데이터와 함수는 어떤 식으로든 
서로 관련이 있다는 것을 금방 알게 됩니다.

특정 데이터에만 의미가 있는 함수가 있다는 사실은, 
데이터와 함수를 개념적으로 묶어 줍니다.

OOP에서는 여전히 전통적인 방식의 데이터와 함수를 사용할 수 있지만, 
OOP는 새로운 구성 요소인 ‘객체(object)’를 도입합니다.

**객체(object)**란 데이터와 함수의 집합입니다.

나중에 이 집합을 어떻게 만드는지 배우게 됩니다.

즉, OOP의 핵심은 데이터와 그 데이터를 처리하는 함수를 하나의 단위(객체)로 묶는 것입니다.
"""
# ---------------------------------------------------------------------------------

# >>>1. 설명
"""

전통적인 데이터와 객체 안의 데이터를 구분하기 위해, **객체 안의 데이터는 ‘속성(attributes)’**이라고 부릅니다.

마찬가지로 전통적인 함수와 객체 안의 함수를 구분하기 위해, **객체 안의 함수는 ‘메서드(methods)’**라고 부릅니다.

따라서 “객체는 데이터와 함수의 집합”이라고 말하기보다는, **“객체는 속성과 메서드의 집합”**이라고 말하는 것이 더 정확합니다.

Python에서는 사실 **거의 모든 것이 객체(object)**입니다!

지금까지 보았던 데이터들은 실제로 특정 타입(type)의 객체입니다. 예: str, int, float

우리가 만든 함수도 다른 타입의 객체입니다.

내장 함수(built-in functions)도 또 다른 타입의 객체입니다.

여기서 “타입(type)” 대신 “클래스(class)”라는 단어를 쓰면, 모든 객체는 어떤 클래스의 구성원(member)이다라는 조금 더 기술적인 표현이 됩니다.

“모든 객체가 클래스의 구성원이다”라는 의미. 객체는 인스턴스이다.

"""
# ---------------------------------------------------------------------------------
# >>>2. 클래스 생성

# [ Attribute ] 
# body에는 클래스 안의 속성과 메서드를 지정하는 문장이 들어간다. 
# 함수 정의와 마찬가지로, 클래스 본문은 반드시 들여쓰기(indent)가 되어 있어야 한다.

# [ Methods ]
# 함수를 클래스 외부에 정의하는 대신, 함수 정의(def)를 클래스 안으로 옮기면 
# 그 함수를 클래스의 메서드(method)로 포함시킬 수 있습니다.
# 이렇게 하면, 해당 메서드는 설계된 객체(object)에서만 작동하도록 보장할 수 있습니다.
# 메서드는 거의 함수와 동일하지만, 두 가지 주목할 만한 차이점이 있습니다.
#   1).하나는 메서드를 정의하는 방식에 영향을 줍니다.
#   2).다른 하나는 메서드를 호출(invoke)하는 방식에 영향을 줍니다.


"""
class <ClassName> :
    <body>

"""

class Patient:
    name = "Jane Doe"
    age = 0
    malady = "healthy"

    def display(self):
        print("Name     = ", self.name)
        print("Age      = ", self.age)
        print("Malady   = ", self.malady)

samuel = Patient()
samuel.name = "Samuel Sneed"
samuel.age = 18
samuel.malady = "broken heart"
samuel.display()

# ---------------------------------------------------------------------------------
# >>>3. dir() 함수 
# Python에서 dir() 함수는 객체가 가지고 있는 속성(attributes)과 
# 메서드(methods) 목록을 보여주는 함수입니다.
# 파이썬에서 우리가 만드는 모든 클래스는 별도로 지정하지 않아도 
# 기본적으로 object라는 최상위 클래스를 상속받습니다. 
# 즉, 직접 정의한 적이 없더라도 파이썬 객체로서 갖춰야 할 기본적인 기능들을 
# 부모(object)로부터 물려받은 것입니다.

# __init__, __str__, __dict__처럼 앞뒤로 밑줄이 두 개씩 붙은 것들을 매직 메서드(Magic Methods) 
# 또는 스페셜 메서드라고 부릅니다. 이들은 파이썬 시스템이 내부적으로 객체를 다루기 위해 미리 약속해둔 도구들입니다.

# __class__: 이 객체가 어떤 클래스에서 만들어졌는지 알려줍니다.
# __dict__: 객체의 속성들이 저장된 사전(dictionary) 데이터를 보여줍니다.
# __str__: print(samuel)을 했을 때 어떤 문자열을 보여줄지 결정합니다.
# __init__: 객체가 생성될 때 초기화하는 역할을 합니다.


type(Patient)
# <class 'type'>

type(samuel)
# <class '__main__.Patient'>

type(samuel.age)
# <class 'int'>

type(samuel.display)
# <class 'method'>

dir(samuel)
# ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', 
#  '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', 
#  '__init__', '__le__', '__lt__', '__module__', '__ne__', '__new__', 
#  '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', 
#  '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'age', 
#  'malady', 'name', 'display']

s= "hello"
dir(s)
# s에 대한 dir이 나온다.
help(s.upper)
# upper 메소드에 대한 설명이 나온다.

# ---------------------------------------------------------------------------------
# >>>4. __init__() 메소드
# 프로그래머가 이 메서드를 정의하면, 객체가 생성될 때 자동으로 호출됩니다.
# 따라서 __init__()는 초기화(initialization) 메서드 역할을 합니다.
# 이 메서드를 사용하면, 객체가 생성될 때 객체의 속성(attributes) 값을 바로 설정할 수 있습니다.
# __init__()를 사용하면 객체 생성 시점에서 속성을 초기화할 수 있습니다.

class Patient:
    def __init__(self, name, age, malady):
        """Initializes the Patient's name, age, and malady."""
        self.name = name
        self.age = age
        self.malady = malady

    def display(self):
        """Display the Patient's attributes."""
        print("Name     = ", self.name)
        print("Age      = ", self.age)
        print("Malady   = ", self.malady)

    def cure(self):
        """Cure the Patient."""
        self.malady = "healthy"

# 객체 생성 및 메서드 실행
sally = Patient("Sally Smith", 21, "bruised ego")
sally.display()

print("\n--- After cure() ---")
sally.cure()
sally.display()


# >>>5. 오퍼레이터 오버로딩
s1 = "Hello" + " World!"  # Concatenation (결합)
print(s1)
# Hello World!

s2 = "&" * 70             # Repetition (반복)
print(s2)
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

10 * "EI" + "-OH!"        # Repetition and concatenation
# 'EIEIEIEIEIEIEIEIEIEI-OH!'

s3 = "#" * 70.0           # TypeError 발생 예시
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: can't multiply sequence by non-int of type 'float'