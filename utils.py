from __future__ import division
import numpy as np


class MiniBatcher(object):
	def __init__(self, N, batch_size=32, loop=True, Y_semi=None, fraction_labelled_per_batch=None):
		self.N = N
		self.batch_size=batch_size
		self.loop = loop
		self.idxs = np.arange(N)
		np.random.shuffle(self.idxs)
		self.curr_idx = 0
		self.fraction_labelled_per_batch = fraction_labelled_per_batch
		if fraction_labelled_per_batch is not None:
			bool_labelled = Y_semi.numpy().sum(1) == 1
			self.labelled_idxs = np.nonzero(bool_labelled)[0]
			self.unlabelled_idxs = np.where(bool_labelled==0)[0]
			np.random.shuffle(self.labelled_idxs)
			np.random.shuffle(self.unlabelled_idxs)
			self.N_labelled = int(self.batch_size*self.fraction_labelled_per_batch)
			self.N_unlabelled = self.batch_size - self.N_labelled
			### check if number of labels are enough, if not repeat labels
			if self.labelled_idxs.shape[0]<self.N_labelled:
				fac = np.ceil(self.N_labelled / self.labelled_idxs.shape[0])
				self.labelled_idxs = self.labelled_idxs.repeat(fac)
		self.start_unlabelled = 0
		self.start_unlabelled_train = 0

	def next(self, train_iter):
		if self.fraction_labelled_per_batch is None:
			if self.curr_idx+self.batch_size >= self.N:
				self.curr_idx=0
				if not self.loop:
					return None
			ret = self.idxs[self.curr_idx:self.curr_idx+self.batch_size]
			self.curr_idx+=self.batch_size
			return ret
		else:
			# WARNING: never terminate (i.e. return None)
			np.random.shuffle(self.labelled_idxs)
			np.random.shuffle(self.unlabelled_idxs)
			return np.array(list(self.labelled_idxs[:self.N_labelled])+list(self.unlabelled_idxs[:self.N_unlabelled]))


# class MiniBatcherPerClass(object):
# 	def __init__(self, N, batch_size=32, Y_semi=None, labels_per_class=None, sample=True, schedule=None, use_sup=None):
# 		self.use_sup = use_sup
# 		self.N = N
# 		self.batch_size=batch_size
# 		self.labels_per_class = int(labels_per_class)
# 		self.idxs_per_class = [[] for i in xrange(Y_semi.size()[1])]
# 		self.unlabelled_idxs = []
# 		self.sample = sample
# 		np_Y_semi = Y_semi.numpy()
# 		for example_idx in xrange(np_Y_semi.shape[0]):
# 			curr_Y = np_Y_semi[example_idx]
# 			if curr_Y.sum() >1:
# 				self.unlabelled_idxs.append(example_idx)
# 			else:
# 				self.idxs_per_class[curr_Y.argmax()].append(example_idx)
# 		## if not enough labels
# 		min_count = np.min([len(labs) for labs in self.idxs_per_class])
# 		if min_count < self.labels_per_class:
# 			fac = np.ceil(self.labels_per_class/min_count)
# 			self.idxs_per_class = [np.array(labs).repeat(fac) for labs in self.idxs_per_class]
# 		self.schedule = schedule
# 		self._make_start_unlabelled()

# 	def _make_start_unlabelled(self):
# 		self.start_unlabelled = len(self.idxs_per_class)*self.labels_per_class
# 		self.start_unlabelled_train = self.start_unlabelled
# 		if self.start_unlabelled == self.batch_size: # completely supervised case..
# 			self.start_unlabelled = 0
		


# 	def next(self, train_iter):
# 		if self.schedule is not None and train_iter == self.schedule[0][0]: 
# 			_, self.labels_per_class = self.schedule.pop(0)
# 			self._make_start_unlabelled()
# 			if self.schedule==[]:
# 				self.schedule=None
# 		if self.sample:
# 			ret_list = []
# 			flag = False
# 			if self.use_sup is not None and self.use_sup>0:
# 				if np.random.rand() < self.use_sup:
# 					flag = True
# 					old_lpc = self.labels_per_class
# 					self.labels_per_class = int(self.batch_size // len(self.idxs_per_class))
# 			for labs in self.idxs_per_class:
# 				np.random.shuffle(labs)
# 				ret_list.append(labs[:self.labels_per_class])
# 			np.random.shuffle(self.unlabelled_idxs)
# 			ret_list.append(self.unlabelled_idxs[:self.batch_size - (len(self.idxs_per_class)*self.labels_per_class)])
# 			if flag:
# 				self.labels_per_class = old_lpc
# 			return np.concatenate(ret_list).astype('int64')
# 		else:
# 			"""
# 			instead of random idxs, loop through them
# 			"""
# 			raise NotImplementedError()