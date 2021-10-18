class RecommenderBase:
    def __init__(self):
        self.data = None
        self.model = None
        self.recos = None
        self.description = None
        self.parameters = None
        self.top_k = None