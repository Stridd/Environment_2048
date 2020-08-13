from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map

cdef extern from "..\Game_2048\Game_2048\Game_2048.cpp":
	pass
	
cdef extern from "..\Game_2048\Game_2048\EpisodeInformation.cpp":
		pass
	
cdef extern from "..\Game_2048\Game_2048\EpisodeInformation.h":
	cdef cppclass EpisodeInformation:
	
		EpisodeInformation() except +
		
		void resetEpisodeInformation()
		void calculateData(int boardSize, vector[vector[int]] board)
		void incrementGameLength()
		void incrementGameScore(int& value)
		void addMove(const int& move)

		int getMaxCell() const
		int getGameScore() const
		int getGameLength() const
		int getEndGameSum() const
		vector[int] getMovesTaken() const

cdef extern from "..\Game_2048\Game_2048\Game_2048.h":
	cdef cppclass Game_2048:
		Game_2048() except +
		Game_2048(const int& x) except +
		
		vector[vector[int]] getBoard() const
		@staticmethod
		vector[int] getAvailableMoves(const vector[vector[int]]& board,
									  const int& boardSize) const 
		
		void setFinishedIfNoActionIsAvailable()
		bool isFinished() const

		void setSeed(const int& seed)

		int sampleAction()
		void takeAction(const int& action)

		void resetGame()

		unordered_map[int, int] getMergedCellsAfterMove()