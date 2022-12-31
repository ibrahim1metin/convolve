import numpy as np
class Convolve:
    def __init__(self,dat1,dat2):
        self.dat1=dat1
        self.dat2=dat2
        self.wn=lambda x : np.exp(-2j*np.pi/x)
    def dft(self,val):
        ln=len(val)
        mat=np.zeros((ln,ln),np.complex64)
        for i in range(ln):
            for j in range(ln):
                mat[i,j]=self.wn(ln)**(i*j)
        return np.matmul(mat,val),mat
    def inverse_dft(self,val):
        ln=len(val)
        mat=np.zeros((ln,ln),np.complex64)
        for i in range(ln):
            for j in range(ln):
                mat[i,j]=(self.wn(ln)**(-i*j))/ln
        return np.matmul(mat,val)
    def convolve(self):
        dat1_dft,_=self.dft(self.dat1)
        dat2_dft,_=self.dft(self.dat2)
        convolution=dat1_dft*dat2_dft
        return np.float32(np.real(self.inverse_dft(convolution))[-1])
print(Convolve((1,1,3),(4,50,6)).convolve())
