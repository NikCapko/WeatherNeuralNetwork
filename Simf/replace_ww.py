l = [
' ',
'Снег непрерывный сильный в срок наблюдения. ', 
'Снег непрерывный слабый в срок наблюдения.  Максимальный диаметр градин составляет 1 мм.', 
'Снег с перерывами слабый в срок наблюдения. ', 
'Снег непрерывный слабый в срок наблюдения. ', 
'Ливневый снег умеренный или сильный в срок наблюдения или за последний час. ', 
'Морось замерзающая слабая. ', 
'Дождь незамерзающий непрерывный слабый в срок наблюдения. ', 
'Ливневый снег слабый в срок наблюдения или за последний час. ', 
'Туман или ледяной туман. ', 
'Туман или ледяной туман, небо видно, без заметного изменения интенсивности в течение последнего часа. ', 
'Дождь (незамерзающий) неливневый. ', 
'Ливневый(ые) дождь(и) слабый(ые) в срок наблюдения или за последний час. ', 
'Снег непрерывный умеренный в срок наблюдения. ', 
'Туман или ледяной туман, небо видно, ослабел за последний час. ', 
'Состояние неба в общем не изменилось. ', 
'Морось (незамерзающая) или снежные зерна неливневые. ', 
'Морось незамерзающая с перерывами слабая в срок наблюдения. ', 
'Гроза, но без осадков, в срок наблюдения. ', 
'Ливневый(ые) дождь(и) со снегом слабый(ые) в срок наблюдения или за последний час. ', 
'Туман или ледяной туман, небо видно, начался или усилился в течение последнего часа. ', 
'Туман или ледяной туман, неба не видно, ослабел за последний час. ', 
'Дождь незамерзающий непрерывный умеренный в срок наблюдения. ', 
'Дождь со снегом или ледяная крупа неливневые. ', 
'Туман или ледяной туман, неба не видно, без заметного изменения интенсивности в течение последнего часа. ', 
'Дымка. ', 
'Дождь или морось со снегом слабые. ', 
'Клочья приземного или ледяного тумана на станции, на море или на суше, высотой не более 2 м над сушей или не более 10 м над морем. ', 
'Снежные зерна (с туманом или без него). ', 
'Туман или ледяной туман, неба не видно, начался или усилился в течение последнего часа. ', 
'Гроза слабая или умеренная с градом в срок наблюдения. ', 
'Замерзающая морось или замерзающий дождь неливневые. ', 
'Морось незамерзающая непрерывная умеренная в срок наблюдения. ', 
'Снег непрерывный слабый в срок наблюдения.  Диаметр смешанного отложения составляет 3 мм.', 
'Гроза слабая или умеренная без града, но с дождем и/или снегом в срок наблюдения. ', 
'Видна молния, грома не слышно. ', 
'Ливневый град, или дождь и град. ', 
'Облака в целом образовывались или развивались. ', 
'Ливневый(ые) дождь(и) со снегом умеренный(ые) или сильный(ые) в срок наблюдения или за последний час. ', 
'Морось замерзающая слабая.  Диаметр отложения при гололеде составляет 2 мм.', 
'Морось незамерзающая непрерывная слабая в срок наблюдения. ', 
'Слабый дождь в срок наблюдения. Гроза в течение последнего часа, но не в срок наблюдения. ', 
'Гроза сильная без града, но с дождем и/или снегом в срок наблюдения. ', 
'Снежные зерна (с туманом или без него).  Диаметр смешанного отложения составляет 0.1 мм.', 
'Ливневая снежная крупа или небольшой град с дождем или без него, или дождь со снегом слабые в срок наблюдения или за последний час. ', 
'Облака в целом рассеиваются или становятся менее развитыми. ', 
'Ливневый(ые) дождь(и) умеренный(ые) или сильный(ые) в срок наблюдения или за последний час. ', 
'Туман или ледяной туман.  Диаметр отложения при гололеде составляет 0.3 мм.', 
'Ливневый снег или ливневый дождь и снег. ', 
'Снег неливневый. ', 
'Дождь незамерзающий с перерывами слабый в срок наблюдения. ', 
'Гроза (с осадками или без них). ', 
'Ливневый(ые) дождь(и). '
]

f_i = open('ww.txt', 'r')
f_o = open('ww_o.txt', 'w')

for line in f_i:
    f = True
    for i in range(len(l)):
        if (line.replace('\n', '') == l[i]):
            f_o.write(str(i) + '\n')
            f = False
            break
    if (f):
        f_o.write(line)
f_o.close()
