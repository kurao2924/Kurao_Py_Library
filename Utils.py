_factorial=[1]
def factorial(n):
    while len(_factorial)<=n:
        _factorial.append((_factorial[-1]*len(_factorial))%mod)
    return _factorial[n]

_inv_factorial=[1]
def inv_factorial(n):
    while len(_inv_factorial)<=n:
        _inv_factorial.append((_inv_factorial[-1]*pow(len(_inv_factorial),mod-2,mod))%mod)
    return _inv_factorial[n]

def binom(n,r):
    if r>=mod:
        raise ValueError("r is too big")
    if n<0:
        return 0
    if r>n:
        return 0
    if r<0:
        return 0
    ans=((factorial(n)*inv_factorial(r))%mod*inv_factorial(n-r))%mod
    return ans

class BinomSum:
    #binom(n,i) i:[l,r)
    def __init__(self,n,l,r):
        self._n=n
        self._l=l
        self._r=r
        self._inv_2=pow(2,mod-2,mod)
        self._sum=0
        for i in range(l,r):
            self._sum+=binom(n,i)
            self._sum%=mod

    def _np(self):
        self._sum*=2
        self._sum-=binom(self._n,self._r-1)
        self._sum+=binom(self._n,self._l-1)
        self._sum%=mod
        self._n+=1
    def _nm(self):
        self._n-=1
        self._sum+=binom(self._n,self._r-1)
        self._sum-=binom(self._n,self._l-1)
        self._sum%=mod
        self._sum*=self._inv_2
        self._sum%=mod
    def _lp(self):
        self._sum-=binom(self._n,self._l)
        self._l+=1
        self._sum%=mod
    def _lm(self):
        self._l-=1
        self._sum+=binom(self._n,self._l)
        self._sum%=mod
    def _rp(self):
        self._sum+=binom(self._n,self._r)
        self._r+=1
        self._sum%=mod
    def _rm(self):
        self._r-=1
        self._sum-=binom(self._n,self._r)
        self._sum%=mod
    def calc(self,n,l,r):
        while n>self._n:
            self._np()
        while n<self._n:
            self._nm()
        while l>self._l:
            self._lp()
        while l<self._l:
            self._lm()
        while r>self._r:
            self._rp()
        while r<self._r:
            self._rm()
        return (self._sum+mod)%mod