import xlrd

class standardReadin:
    def __init__(self,filename,islabel):
        self.texfile=1
        data = xlrd.open_workbook(filename)
        textfile=open("SmartStoplist.txt",'r')
        stlist=textfile.readlines()
        textfile.close()
        self.stoplist =[word.strip() for word in stlist]
        self.table = data.sheets()[0]
        self.nrows = self.table.nrows  # 行数
        self.ncols = self.table.ncols  # 列数
        print("find " ,self.nrows-1, "CE documents, index from 1 to ",self.nrows-1)
        self.islabel=islabel
        if islabel:
            self.map = data.sheets()[1]
            self.kwlist={}
            for i in range(1,self.map.nrows):
                # print(self.map.cell(i,2).value)
                try:id= int(self.map.cell(i,2).value)
                except ValueError:
                    continue
                if (self.kwlist.get(id)):
                    newkw=self.kwlist.get(id)
                    newkw.append(self.map.cell(i,1).value)
                else:
                    newkw=[self.map.cell(i,1).value]
                self.kwlist[id]=newkw




    def readin(self):

        cell_A1 = self. table.cell(5, 3).value

        print (cell_A1)

    def getFilename(self,index):
        name=self.table.cell(index, 0).value
        #name = "".join(name.split())
        return name
    def getTitle(self,index):
        name = self.table.cell(index, 3).value
        #name = "".join(name.split())
        return name
    def getBrief(self,index):
        name = self.table.cell(index, 4).value
        #name = "".join(name.split())
        return name

    def loadkw(self,index):
        if self.islabel==False:
            return []
        else:
            try:id=int(self.table.cell(index,5).value)
            except ValueError:
                return []
            return self.kwlist[id]

    def getsize(self):
        return self.nrows-1


    def writekw(self):
        return 0


# St=standardReadin("Scopes.xlsx",True)
# print(St.stoplist[53])
# kw=St.loadkw(3)
# print(kw)