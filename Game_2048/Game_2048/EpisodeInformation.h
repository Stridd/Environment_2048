#pragma once
#include <vector>
class EpisodeInformation
{
	private:
		int maxCell;
		int gameScore;
		int gameLength;
		int endGameSum;
		std::vector<int> movesTaken;

	public:
		EpisodeInformation();
		void resetEpisodeInformation();
		void calculateData(int& boardSize, std::vector<std::vector<int>>& board);
		void incrementGameLength();
		void incrementGameScore(int& value);
		void addMove(const int& move);

		int getMaxCell() const;
		int getGameScore() const;
		int getGameLength() const;
		int getEndGameSum() const;
		std::vector<int> getMovesTaken() const;
		
};

