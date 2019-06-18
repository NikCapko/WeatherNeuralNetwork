l = [
'nan',
'70 – 80%.',
'100%.',
'20–30%.',
'Облаков нет.',
'60%.',
'Небо не видно из-за тумана и/или других метеорологических явлений.',
'40%.',
'90  или более, но не 100%',
'10%  или менее, но не 0',
'50%.'
]

k = [0, 75, 100, 25, 0, 60, 100, 40, 95, 5, 50]

f_i = open('n.txt', 'r')
f_o = open('n_o.txt', 'w')

for line in f_i:
    f = True
    for i in range(len(l)):
        if (line.replace('\n', '') == l[i]):
            f_o.write(str(k[i]) + '\n')
            f = False
            break
    if (f):
        f_o.write(line)
f_o.close()
