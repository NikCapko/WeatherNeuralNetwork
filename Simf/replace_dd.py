l = [
'Ветер, дующий с севера',
'Ветер, дующий с северо-северо-востока',
'Ветер, дующий с северо-востока',
'Ветер, дующий с востоко-северо-востока',
'Ветер, дующий с востока',
'Ветер, дующий с востоко-юго-востока',
'Ветер, дующий с юго-востока',
'Ветер, дующий с юго-юго-востока',
'Ветер, дующий с юга',
'Ветер, дующий с юго-юго-запада',
'Ветер, дующий с юго-запада',
'Ветер, дующий с западо-юго-запада'
'Ветер, дующий с запада',
'Ветер, дующий с западо-северо-запада',
'Ветер, дующий с северо-запада',
'Ветер, дующий с северо-северо-запада',
'Переменное направление',
'Штиль, безветрие'
]

f_i = open('dd.txt', 'r')
f_o = open('dd_o.txt', 'w')

for line in f_i:
    f = True
    for i in range(len(l)):
        if (line.replace('\n', '') == l[i]):
            f_o.write(str(i*2) + '\n')
            f = False
            break
    if (f):
        f_o.write(line)
f_o.close()
