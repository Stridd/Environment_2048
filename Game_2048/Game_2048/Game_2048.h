#pragma once

#include <random>
#include <set>
#include <tuple>

using fromToStopStep = std::tuple<int, int, int, int>;
using boardType = std::vector<std::vector<int> >;

class Game_2048
{
	private:

		static int currentEpisode;

		int boardSize;

		std::mt19937 RNG;
		boardType board;

		fromToStopStep getIterationElementsByDirection(const int&);

		void addTwoTiles();
		void addRandomTile();

		std::vector<std::pair<int, int>> getEmptyPositions();
		void assignValueToRandomEmptyCell(std::vector<std::pair<int, int>>& emptyTiles);

		void move(const int& , const int& );
		void resetBoard();

		bool isGameFinished;

	public:

		Game_2048();
		Game_2048(const unsigned int&);

		boardType getBoard() const;
		static std::vector<int> getAvailableMoves(const boardType& board, const int& boardSize);

		void setSeed(const int& seed);
		
		void setFinishedIfNoActionIsAvailable();
		bool isFinished() const;

		int sampleAction();
		void takeAction(const int& action);

		void resetGame();
		void printBoard();

		void setBoard(const boardType& board);

		enum Moves
		{
			UP,
			RIGHT,
			DOWN,
			LEFT
		};
};
