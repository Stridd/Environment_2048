#include "Game_2048.h"


int Game_2048::currentEpisode = 0;

Game_2048::Game_2048(const unsigned int& boardSize)
{
	this->boardSize = boardSize;
	this->board = std::vector<std::vector<int>> 
					  (boardSize, std::vector<int>(boardSize, 0));

	currentEpisode = 0;
	collector.emplace_back();

	RNG.seed(std::random_device()());

	addTile();
	addTile();
}

Game_2048::Game_2048()
{
	this->boardSize = 0;

	this->board = std::vector<std::vector<int>>
		(boardSize, std::vector<int>(boardSize, 0));

	currentEpisode = 0;
	collector.emplace_back();
}

void Game_2048::resetBoard()
{
	for (int i = 0; i < boardSize; ++i)
		for (int j = 0; j < boardSize; ++j)
			board[i][j] = 0;
		
	addTile();
	addTile();
}

std::vector<std::vector<int> > Game_2048::getBoard() const
{
	return board;
}

void Game_2048::resetGame()
{
	resetBoard();
	currentEpisode += 1;
	collector.emplace_back();
}

void Game_2048::calculateEndGameData()
{
	collector[currentEpisode].calculateData(boardSize, board);
}

int Game_2048::getScore()
{
	return collector[currentEpisode].getGameScore();
}

std::vector<EpisodeInformation> Game_2048::getEpisodesData()
{
	return collector;
}

EpisodeInformation Game_2048::getCurrentEpisodeData()
{
	return collector[currentEpisode];
}

std::vector<int> Game_2048::getAvailableMoves(const std::vector<std::vector<int> >& board,
											  const int& boardSize) const
{
	std::set<int> availableMoves;

	for(int i = 0; i < boardSize; ++i)
		for (int j = 0; j < boardSize; ++j)
		{
			if (i > 0)
				if (board[i - 1][j] == board[i][j] || board[i - 1][j] == 0)
					availableMoves.insert(UP);

			if (j < boardSize - 1)
				if (board[i][j + 1] == 0 || board[i][j + 1] == board[i][j])
					availableMoves.insert(RIGHT);

			if(i < boardSize - 1)
				if (board[i + 1][j] == board[i][j] || board[i + 1][j] == 0)
					availableMoves.insert(DOWN);

			if (j > 0)
				if (board[i][j - 1] == 0 || board[i][j - 1] == board[i][j])
					availableMoves.insert(LEFT);
		}

	return std::vector<int>(availableMoves.begin(), availableMoves.end());
}

bool Game_2048::isFinished() const
{
	return getAvailableMoves(board, boardSize).size() == 0;
}

int Game_2048::sampleAction()
{
	std::vector<int> availableMoves(getAvailableMoves(board, boardSize));

	std::uniform_int_distribution<int> indexesAvailable(0, availableMoves.size() - 1);

	int index = indexesAvailable(RNG);

	return availableMoves[index];
}

void Game_2048::takeAction(const int& action)
{
	switch (action)
	{
		case UP:
			move(-1, 0);
			break;
		case RIGHT:
			move(0, +1);
			break;
		case DOWN:
			move(+1, 0);
			break;
		case LEFT:
			move(0, -1);
			break;
	}
	collector[currentEpisode].addMove(action);
	collector[currentEpisode].incrementGameLength();
	addTile();
}

void Game_2048::addTile()
{
	std::vector<std::pair<int, int> > emptyTiles;

	for(int i = 0; i != boardSize; ++i)
		for (int j = 0; j != boardSize; ++j)
			if (board[i][j] == 0)
				emptyTiles.push_back(std::make_pair(i, j));

	if (emptyTiles.size() != 0)
	{
		std::uniform_int_distribution<int> emptyCellDist(0, emptyTiles.size() - 1);

		std::pair<int, int> position = emptyTiles[emptyCellDist(RNG)];

		std::uniform_real_distribution<double> valueDistribution(0, 1.0);

		double probability = valueDistribution(RNG);

		board[position.first][position.second] = probability >= 0.9 ? 4 : 2;
	}

}

auto Game_2048::addRanges(const int& direction)
{
	const int start = direction == 0 ? 0 : (direction == -1 ? 1 : boardSize - 1);
	const int end = direction == 0 ? boardSize : (direction == -1 ? boardSize : -1);
	const int stop = boardSize + end * direction;
	const int step = direction == 0 ? -1 : direction;

	return std::make_tuple(start, end, stop, step);
}

void Game_2048::move(const int& yDirection, const int& xDirection)
{
	std::vector< std::vector<bool> > cellWasCombined(boardSize, std::vector<bool>(boardSize, false));

	int startX, endX, stopX, stepX;
	int startY, endY, stopY, stepY;

	std::tie(startX, endX, stopX, stepX) = addRanges(xDirection);
	std::tie(startY, endY, stopY, stepY) = addRanges(yDirection);

	for (int i = startY; i != endY; i += -stepY)
		for (int j = startX; j != endX; j += -stepX)
		{
			int line = i;
			int column = j;

			while (line != stopY &&
				   column != stopX &&
				   board[line][column] != 0
				  )
			{
				if (board[line][column] == board[line + yDirection][column + xDirection] &&
					!cellWasCombined[line][column] &&
					!cellWasCombined[line + yDirection][column + xDirection])
				{
					board[line + yDirection][column + xDirection] *= 2;
					board[line][column] = 0;
					cellWasCombined[line + yDirection][column + xDirection] = true;

					collector[currentEpisode].incrementGameScore(board[line + yDirection][column + xDirection]);
				}
				else
					if (board[line + yDirection][column + xDirection] == 0)
					{
						board[line + yDirection][column + xDirection] = board[line][column];
						board[line][column] = 0;
					}

				line += yDirection;
				column += xDirection;
			}
		}
}