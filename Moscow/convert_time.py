f = open("time.txt", "r")
f2 = open("ti.txt", "w")
for line in f:
    if (line != ""):
        s = line.split(" ")[1].split(':')[0]
        f2.write(str(int(s)) + "\n")
f2.close()
