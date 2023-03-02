# This file holds the mean and standard deviation for each task
class Norm:
    def __init__(self, name):
        if name == "solar":
            self.mean = [0.507, 0.513, 0.462]
            self.std = [0.172, 0.133, 0.114]
        elif name == "crop-delineation":
            self.mean = [0.238, 0.297, 0.317]
            self.std = [0.187, 0.123, 0.114]
        elif name == "building":
            self.mean = [0.406, 0.428, 0.394]
            self.std = [0.201, 0.183, 0.176]
        elif name == "climate+":
            self.mean = [0.460, 0.440, 0.378]
            self.std = [0.179, 0.139, 0.123]
        elif name == "eurosat":
            self.mean = [0.344, 0.380, 0.408],
            self.std = [0.204, 0.137, 0.115]
        else:
            raise NotImplementedError("Task not implemented")
