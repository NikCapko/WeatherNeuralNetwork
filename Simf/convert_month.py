f = open("m.txt", "r")
f2 = open("mi.txt", "w")
for line in f:
    if (line != ""):
        s = line.split(".")[1]
        f2.write(str(int(s)) + "\n")
f2.close()
