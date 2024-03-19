class KNN:
    def __init__(self):
        self.x = None
        self.y = None

    def train(self,x,y):
        self.x = x
        self.y = y

    def predict(self,x,k):
        distance_label = [
            (self.distance(x,train_point),train_label)
            for train_point , train_label in zip(self.x,self.y)
        ]
        neighbors = sorted(distance_label)[:k]
        return sum(labels for _,labels in neighbors)/k
