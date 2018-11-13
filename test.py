import numpy as np

def makematrix(n,m):
    matrix=np.zeros((n,m))
    flag=True
    x=0
    y=0
    count=1
    matrix[x][y]=count;
    while(1):
        if(flag):
            if(x+1>=n): break
            if(y-1>-1):
                x=x+1
                y=y-1
            else:
                x=x+1
                flag=False
        else:
            if(y+1>=m): break
            if(x-1>-1 ):
                x=x-1
                y=y+1
            else:
                y=y+1
                flag=True
        count=count+1
        matrix[x][y]=count
    #end of while
    count=20
    x=n-1
    y=m-1
    matrix[x][y]=count;
    flag=False
    while(1):
      if(flag):
        if(x-1<0): break
        if(y+1<n):
          x=x-1
          y=y+1
         else:
          x=x-1
          flag=false
       else:
         if(y-1<0): break
         if(x+1<m):
          x=x+1
          y=y-1
         else:
          y=y-1
          flag=true
       count--;
       matrix[x][y]=count
       #end of while
   return matrix

def main():
m=4,n=5
makematrix(m,n)
