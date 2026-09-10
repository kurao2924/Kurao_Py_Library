class Binary_Trie:
    def __init__(self,max_bit: int = 31):
        """
        data in [0,n]
        """
        self.cnt=[0]
        self.xor=0
        self.left=[-1]
        self.right=[-1]
        self.max_bit=max_bit

    def add(self,x,cnt=1):
        x^=self.xor
        now=0
        for i in range(self.max_bit-1,-1,-1):
            bit=int((1<<i)&x>0)
            if self.left[now]==-1:
                self.left[now]=len(self.cnt)
                self.right[now]=len(self.cnt)+1
                for _ in range(2):
                    self.cnt.append(0)
                    self.left.append(-1)
                    self.right.append(-1)
            self.cnt[now]+=cnt
            if bit==0:
                now=self.left[now]
            else:
                now=self.right[now]
        self.cnt[now]+=1

    def find(self,x):
        now=0
        x^=self.xor
        for i in range(self.max_bit-1,-1,-1):
            bit=int((1<<i)&x>0)
            if self.cnt[now]==0:
                return 0
            else:
                if bit==0:
                    now=self.left[now]
                else:
                    now=self.right[now]
        return self.cnt[now]
    
    def delete(self,x,cnt=1):
        now=0
        x^=self.xor
        if self.find(x)<cnt:
            return
        for i in range(self.max_bit-1,-1,-1):
            bit=int((1<<i)&x>0)
            self.cnt[now]-=cnt
            if bit==0:
                now=self.left[now]
            else:
                now=self.right[now]
        self.cnt[now]-=cnt

    def bisect_left(self,x):
        now=0
        ans=0
        for i in range(self.max_bit-1,-1,-1):
            if self.xor&(1<<i)==0:
                if x&(1<<i)>0:
                    if self.left[now]!=-1:
                        ans+=self.cnt[self.left[now]]
                    now=self.right[now]
                else:
                    now=self.left[now]
            else:
                if x&(1<<i)>0:
                    if self.right[now]!=-1:
                        ans+=self.cnt[self.right[now]]
                    now=self.left[now]
                else:
                    now=self.right[now]
            if now==-1:
                return ans
        return ans

    def __getitem__(self,k):
        if k<0:
            k+=len(self)
        if not 0<=k<self.cnt[0]:
            raise ValueError("Binary_Trie index out of range")
        now=0
        ans=0
        for i in range(self.max_bit-1,-1,-1):
            ans*=2
            if self.xor&(1<<i)==0:
                if k-self.cnt[self.left[now]]>=0:
                    k-=self.cnt[self.left[now]]
                    now=self.right[now]
                    ans+=1
                else:
                    now=self.left[now]
            else:
                if k-self.cnt[self.right[now]]>=0:
                    k-=self.cnt[self.right[now]]
                    now=self.left[now]
                    ans+=1
                else:
                    now=self.right[now]
        return ans
    
    def bisect_right(self,x):
        return self.bisect_left(x)+self.find(x)
    
    def __len__(self):
        return self.cnt[0]
    
    def __repr__(self):
        return f"Binary_Trie({[self[j] for j in range(len(self))]})"
    
    def xor_all(self,x):
        self.xor^=x
