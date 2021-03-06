# breakpoint()
# 디버깅을 목적으로 중단점을 만드는 함수 
# 중단점을 사용하면 코드에 에러가 발생할 때 매우 편리하게 사용할 수 있다. 
import pdb 

pdb.set_trace()

print('abc')

# 위와 동일한 함수.
breakpoint()

print('abc')



# exec()

# 문자열로 표현된 function을 인수로 받아 파이썬 컴파일 코드로 변환.

# exec(object[, globals[,locals]])

a = 5
exec('a = a + 4')
a

exec("one = 1 + 9 + pow(2, 2)")
one
# 14

kk = {"pw" : pow, 'a' : 0}
exec('a = pw(3, 3)', kk)
kk['a']
# 27



# isinstance()
# 자료형 확인 사용을 안할 것 같음..

isinstance(1.2, int)
# False

isinstance(1.2, float)
# True

isinstance('hi', str)
# True 



# ord()
# ord(c)는 문자의 유니코드 값을 돌려주는 함수

ord('a')
# 96

ord('3')
# 51



# sum() / sum(iterable)
# 가장 많이 사용하는 함수이지 않을까 한다.
# numpy 에서도 sum function이 있다. 
# 입력받은 list나 tuple의 모든 요소 합을 반환해준다. 
# DataFrame일 경우 axis 를 지정해주어야 한다. 
sum([1, 2, 3, 4, 5]) # 15



# bytearray()
# 1byte 단위 값을 저장하는 sequence type 

bytes(10)    # 0이 10개 들어있는 바이트 객체 생성
b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
bytes([10, 20, 30, 40, 50])    # 리스트로 바이트 객체 생성
b'\n\x14\x1e(2'
bytes(b'hello')    # 바이트 객체로 바이트 객체 생성
b'hello'


# filter() 해당 하는 값을 걸러준다.

target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_even(n):
    return True if n % 2 == 0 else False

result = filter(is_even, target)

print(list(result))


target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = filter(lambda x : x%2==0, target)
print(list(result))


# issubclass()
# Param class가 param classinfo의 sub class가 맞으면 True를 리턴

class fooboo(object):
    def __init__(self, val):
        self.val = val 
    
class gooboo(fooboo):
    def __init__(self, val, val2):
        self.val = val 
        self.val2 = val2 

issubclass(gooboo, fooboo)

issubclass(fooboo, gooboo)

issubclass(fooboo, object)


# pow()
2 ** 3 # 8
pow(2, 3) # 8

pow(100, 2) # 10000

pow(100, -2) # 0.0001

pow(100, (1/2)) # 10.0

pow(100, -1/2) # 0.1

pow(100, -1/3) # 0.21544---


# 나눈 나머지
pow(2, 3) % 3 # 2

pow(2, 3, 3) # 2

# super()
# 상속, 오버라이딩
# 자식 클래스에서 부모클래스의 내용을 사용하고 싶을때 사용

class father():  # 부모 클래스
    def handsome(self):
        print("잘생겼다")
 
 
class brother(father):  # 자식클래스(부모클래스) 아빠매소드를 상속받겠다
    '''아들'''
 
 
class sister(father):  # 자식클래스(부모클래스) 아빠매소드를 상속받겠다
    def pretty(self):
        print("예쁘다")
 
    def handsome(self):
       super().handsome()
 
 
brother = brother()
brother.handsome()
 
girl = sister()
girl.handsome()
girl.pretty()




class father():  # 부모 클래스
    def __init__(self, who):
        self.who = who
 
    def handsome(self):
        print("{}를 닮아 잘생겼다".format(self.who))
 
class sister(father):  # 자식클래스(부모클래스) 아빠매소드를 상속받겠다
    def __init__(self, who, where):
        super().__init__(who)
        self.where = where
 
    def choice(self):
        print("{} 말이야".format(self.where))
 
    def handsome(self):
       super().handsome()
       self.choice()
 
girl = sister("아빠", "얼굴")
girl.handsome()

