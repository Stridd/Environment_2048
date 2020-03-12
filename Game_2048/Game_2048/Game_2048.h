#pragma once

#include <random>
#include <set>
#include <tuple>
#include "EpisodeInformation.h"

class Game_2048
{
	private:

		std::vector<EpisodeInformation> collector;
		static int currentEpisode;

		int boardSize;

		std::mt19937 RNG;
		std::vector<std::vector<int> > board;
		
		void resetBoard();
		void addTile();
		void move(const int& , const int& );
		auto addRanges(const int&);

		enum Moves 
		{
			UP,
			RIGHT,
			DOWN,
			LEFT
		};

	public:
		Game_2048(const unsigned int&);
		Game_2048();

		std::vector<std::vector<int> > getBoard() const;
		std::vector<int> getAvailableMoves(const std::vector<std::vector<int> >& board, const int& boardSize) const;

		bool isFinished() const;
		int sampleAction();

		void takeAction(const int& action);
		void resetGame();
		void calculateEndGameData();
		int getScore();
		std::vector<EpisodeInformation> getEpisodesData();
		EpisodeInformation getCurrentEpisodeData();

};
