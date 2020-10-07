from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map

cdef extern from "..\Game_2048\Game_2048\Game_2048.cpp":
	pass

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