# Simple Early Stopping class
class EarlyStopping():
	def __init__(self, patience=10, min_delta=0.0) -> None:
		self.patience = patience
		self.min_delta = min_delta
		self.best_loss = 1_000_000
		self.counter = 0
	
	def check(self, current_loss) -> bool:
		if current_loss + self.min_delta > self.best_loss:
			self.counter += 1
			if self.counter >= self.patience:
				return True
		else:
			self.counter = 0
			self.best_loss = current_loss
		return False
	
	def setBestLoss(self, current_loss):
		self.best_loss = current_loss