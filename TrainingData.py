class TrainingData():
    def __init__(self):
        self.X = DataPair()
        self.Y = DataPair()
        self.X.train, self.Y.train, self.X.test, self.Y.test = get_binary_data()



class DataPair():
    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, train):
        self._train = train

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, test):
        self._test = test