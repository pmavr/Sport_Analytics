class LRPolicy(object):
    def __init__(self, epoch_count=1, niter=100, niter_decay=100):
        self.epoch_count = epoch_count
        self.niter = niter
        self.niter_decay = niter_decay

    def __call__(self, epoch):
        lr_l = 1.0 - max(0, epoch + 1 + self.epoch_count - self.niter) / float(self.niter_decay + 1)
        return lr_l