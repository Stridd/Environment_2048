# distutils: language = c++

from Game_2048 cimport Game_2048

from libcpp.vector cimport vector

import numpy as np

cdef class Environment_2048:
	cdef Game_2048 game
	
	def __cinit__(self, int x):
		self.game = Game_2048(x)
	
	def getBoard(self):
		return self.game.getBoard()
		
	def sampleAction(self):
		return self.game.sampleAction()
		
	def resetGame(self):
		return self.game.resetGame()
		
	def takeAction(self, int x):
		return self.game.takeAction(x)
		
	def isFinished(self):
		return self.game.isFinished()	
		
	def getAvailableMoves(self, board, boardSize):
		return Game_2048.getAvailableMoves(board, boardSize)

	def setSeed(self, seed):
		self.game.setSeed(seed)

	def setFinishedIfNoActionIsAvailable(self):
		self.game.setFinishedIfNoActionIsAvailable()