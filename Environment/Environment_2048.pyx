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
		
	def calculateEndGameData(self):
		return self.game.calculateEndGameData()
	
	def getScore(self):
		return self.game.getScore()
		
	def getAvailableMoves(self, board, boardSize):
		return self.game.getAvailableMoves(board, boardSize)
	
	def getEpisodesData(self):
	
		gameLengths = []
		gameScores = []
		gameEndGameSums = []
		gameMaxCells = []
		
		episodesInfo = self.game.getEpisodesData()
		
		cdef int i
		for i in range(episodesInfo.size()):
			gameLengths.append(episodesInfo[i].getGameLength())
			gameScores.append(episodesInfo[i].getGameScore())
			gameEndGameSums.append(episodesInfo[i].getEndGameSum())
			gameMaxCells.append(episodesInfo[i].getMaxCell())
		
		gameLengths = np.array(gameLengths)
		gameScores = np.array(gameScores)
		gameEndGameSums = np.array(gameEndGameSums)
		gameMaxCells = np.array(gameMaxCells)
		return gameLengths, gameScores, gameEndGameSums, gameMaxCells
		
	def getCurrentEpisodeData(self):
	
		episodeInformation = self.game.getCurrentEpisodeData()
		
		gameLength = episodeInformation.getGameLength()
		gameScore = episodeInformation.getGameScore()
		gameEndGameSum = episodeInformation.getEndGameSum()
		gameMaxCell = episodeInformation.getMaxCell()
		
		return gameLength, gameScore, gameEndGameSum, gameMaxCell
		
	
	
	