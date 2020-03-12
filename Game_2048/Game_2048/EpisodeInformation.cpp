#include "EpisodeInformation.h"

EpisodeInformation::EpisodeInformation()
{
	resetEpisodeInformation();
}

void EpisodeInformation::resetEpisodeInformation()
{
	gameScore = 0;
	gameLength = 0;
	endGameSum = 0;
	maxCell = 0;
	movesTaken.clear();
}

void EpisodeInformation::calculateData(int& boardSize, std::vector<std::vector<int> >& board)
{
	for(int i = 0; i < boardSize; ++i)
		for (int j = 0; j < boardSize; ++j)
		{
			endGameSum += board[i][j];
			if (maxCell < board[i][j])
				maxCell = board[i][j];
		}
}

void EpisodeInformation::incrementGameLength()
{
	gameLength++;
}

void EpisodeInformation::incrementGameScore(int& value)
{
	gameScore += value;
}

void EpisodeInformation::addMove(const int& move)
{
	movesTaken.push_back(move);
}

int EpisodeInformation::getMaxCell() const
{
	return maxCell;
}

int EpisodeInformation::getGameScore() const
{
	return gameScore;
}

int EpisodeInformation::getGameLength() const
{
	return gameLength;
}

int EpisodeInformation::getEndGameSum() const
{
	return endGameSum;
}

std::vector<int> EpisodeInformation::getMovesTaken() const
{
	return movesTaken;
}
