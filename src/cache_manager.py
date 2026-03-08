import numpy as np

def cosine_similarity(a,b):

    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

class CacheManager:

    def __init__(self,threshold=0.85):

        self.cache={}

        self.threshold=threshold

        self.hits=0
        self.misses=0

    def lookup(self,embedding,cluster):

        if cluster not in self.cache:

            self.misses+=1
            return False,None,-1.0

        best=None
        best_score=-1.0

        for entry in self.cache[cluster]:

            sim=cosine_similarity(embedding,entry["embedding"])

            if sim>best_score:

                best_score=sim
                best=entry


        if best_score>=self.threshold:

            self.hits+=1
            return True,best,best_score

        self.misses+=1

        return False,None,best_score


    def store(self,query,embedding,result,cluster):

        if cluster not in self.cache:

            self.cache[cluster]=[]

        self.cache[cluster].append({

            "query":query,
            "embedding":embedding,
            "result":result
        })

    def stats(self):

        total=sum(len(v) for v in self.cache.values())

        rate=0

        if self.hits+self.misses>0:

            rate=self.hits/(self.hits+self.misses)

        return{

            "total_entries":total,
            "hit_count":self.hits,
            "miss_count":self.misses,
            "hit_rate":rate
        }


    def clear(self):

        self.cache={}
        self.hits=0
        self.misses=0