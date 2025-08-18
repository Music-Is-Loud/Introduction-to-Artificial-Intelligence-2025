import numpy as np
tmpLayoutText=[[' 'for i in range(5)]for j in range(11)]
for j in range(11):
    for i in range(5):
        
        
        print(tmpLayoutText[j][i],end='!')
    
LayoutText=[]
for line in tmpLayoutText:
    tmpstr=""
    for lineChar in line:
        tmpstr+=lineChar
    LayoutText.append(tmpstr)
for i in LayoutText:
    print(i)