import csv
def read_data(findS):
    with open(findS+'.csv','r') as csvfile:
        datareader = csv.reader(csvfile,delimiter=',')
        traindata=[]
        for row in datareader:
            traindata.append(row)
        for da in traindata[1:]:
            print(da)
           
    return (traindata)
h=['phi','phi','phi','phi','phi','phi']
data=read_data('D:\ML Practical/FindS')
def isConsistent(h,d):
    if len(h)!=len(d)-1:
        print('Number of attribute are not same in hypothesis.')
        return False
    else:
        matched=0
        for i in range(len(h)):
            if((h[i]==d[i] or h[i]=='any')):
                matched=matched+1
        if(matched==len(h)):
            return True
        else:
            return False
def makeConsistent(h,d):
    for i in range(len(h)):
        if(h[i]==d[i] or h[i]=='phi'):
            h[i]=d[i]
        elif(h[i]!=d[i]):
            h[i]='any'
    return h
print('Begin : Hypothesis:',h)
print('==================================================')
for d in data[1:]:
    # print(d)
    if d[len(d)-1]=='yes':
        if( isConsistent(h,d)):
            pass
        else:
            h=makeConsistent(h,d)
        print('Training data : ',d)
        print('Updated Hypothesis : ',h)
        print()
        print('-------------------------------')
print('========================================')
print('maximally specific data set End : Hypothesis :',h)