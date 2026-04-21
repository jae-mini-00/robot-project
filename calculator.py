def add(a, b): return a + b

def min(a, b): return a- b

def mul(a, b): return a * b

def div(a, b):
    if b == 0:
        return print("0으로는 나눌수 없습니다.")
    return a / b

def calculator():
    fnc = {'+':add, '-': min, '*':mul, '/':div}

    while True:
        try:
            op = input("연산자를 입력하세요(+, -, *, /, q):")

            if op == 'q':
                break
            elif op not in ['+', '-', '*', '/']:
                continue

            num1 = float(input("숫자 1(M):"))
            num2 = float(input("숫자 2(M):"))
            print(fnc[op](num1, num2))
        except ValueError as e:
            print(e)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    calculator()