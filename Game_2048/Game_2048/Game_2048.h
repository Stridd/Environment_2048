#pragma once

#include <random>
#include <set>
#include <tuple>
#include <unordered_map>

using fromToStopStep = std::tuple<int, int, int, int>;
using boardType = std::vector<std::vector<int> >;
using emptyPositionsVector = std::vector<std::pair<int, int>>;

class Game_2048
{
	private:

		bool isGameFinished;

		int boardSize;
		boardType board;

		std::mt19937 RNG;
		
		std::unordered_map<int, int> mergedCellsAfterMove;

		fromToStopStep getIterationElementsByDirection(const int&);

		void addTwoTiles();
		void addRandomTile();

		emptyPositionsVector getEmptyPositions();
		void assignValueToRandomEmptyCell(emptyPositionsVector& emptyTiles);

		void move(const int& , const int& );
		void resetBoard();

		bool canBeMergedAtPositions(const int&, const int&, const int&, const int&, std::vector< std::vector<bool> >&);
		
		void storeMergedCellsInformation(int& value);
		void emptyMergedCellsInformation();

	public:

		Game_2048();
		Game_2048(const unsigned int&);
		Game_2048(const unsigned int&, boardType& );

		boardType getBoard() const;
		static std::vector<int> getAvailableMoves(const boardType& board, const int& boardSize);

		std::unordered_map<int, int> getMergedCellsAfterMove();

		void setSeed(const int& seed);
		
		void setFinishedIfNoActionIsAvailable();
		bool isFinished() const;

		int sampleAction();
		void takeAction(const int& action);

		void resetGame();
		void printBoard();

		enum Moves
		{
			UP,
			RIGHT,
			DOWN,
			LEFT
		};
};
