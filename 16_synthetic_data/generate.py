
import random
def generate(n=100):
    words=["cat","dog","tree","runs","blue"]
    return [" ".join(random.choices(words,k=5)) for _ in range(n)]
