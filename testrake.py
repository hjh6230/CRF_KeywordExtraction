import RAKE as rake
from readin import standardReadin as SR
import csv
import operator

rake_object = rake.Rake("SmartStoplist.txt")
# sample_file = open("testtext.txt", 'r')
# text = sample_file.read()

CEs=SR("Scopes.xlsx")
# numCE=SR.nrows-1;

with open('keywords.csv','w') as f:
    f_csv = csv.writer(f)

    for index in range(1, 10):
        text = CEs.getBrief(index)
        keywords = rake_object.run(text)
        kwList=[]
        name=CEs.getFilename(index)
        name="".join(name.split())
        kwList.append(name)
        maxKw=8;
        count=0
        for kw in keywords:
            kwList.append(kw[0])
            count=count+1
            # if (count>maxKw):
            #     break
        print(kwList)
        f_csv.writerow(kwList)


