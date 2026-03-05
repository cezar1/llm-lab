
class KVCache:
    def __init__(self): self.k=[]; self.v=[]
    def add(self,k,v): self.k.append(k); self.v.append(v)
    def get(self): return self.k,self.v
