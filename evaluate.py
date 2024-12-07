import basic

mian = basic.MainGo()
info = []
while True:
    info.append(input("input exit to quit."))
    if info[-1] == "exit":
        info.pop()
        break

mian.input_email(info)
mian.process()
ans = mian.evaluate()
print("\n\n")
print(ans)