#pragma once

#include <random>
#include <set>
#include <tuple>

using fromToStopStep = std::tuple<int, int, int, int>;

class Game_2048
{
	private:

		static int currentEpisode;

		int boardSize;

		std::mt19937 RNG;
		std::vector<std::vector<int> > board;

		fromToStopStep getIterationElementsByDirection(const int&);

		void addTwoTiles();
		void addRandomTile();

		std::vector<std::pair<int, int>> getEmptyPositions();
		void assignValueToRandomEmptyCell(std::vector<std::pair<int, int>>& emptyTiles);

		void move(const int& , const int& );
		void resetBoard();

		bool isGameFinished;

		enum Moves 
		{
			UP,
			RIGHT,
			DOWN,
			LEFT
		};

	public:
		Game_2048();
		Game_2048(const unsigned int&);

		std::vector<std::vector<int> > getBoard() const;
		std::vector<int> getAvailableMoves(const std::vector<std::vector<int> >& board, const int& boardSize) const;

		bool isFinished() const;

		void setSeed(const int& seed);

		void setFinished();

		int sampleAction();
		void takeAction(const int& action);

		void resetGame();
		void printBoard();
};
