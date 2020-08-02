#include "Game_2048.h"
#include <iostream>

int Game_2048::currentEpisode = 0;

Game_2048::Game_2048()
{
	this->boardSize = 0;

	this->board = boardType
		(boardSize, std::vector<int>(boardSize, 0));

	currentEpisode = 0;
}

Game_2048::Game_2048(const unsigned int& boardSize)
{
	this->boardSize = boardSize;
	this->board = boardType
					  (boardSize, std::vector<int>(boardSize, 0));

	currentEpisode = 0;
	isGameFinished = false;
	RNG.seed(std::random_device()());
	addTwoTiles();
}

void Game_2048::resetGame()
{
	resetBoard();
	isGameFinished = false;
	currentEpisode += 1;

}

void Game_2048::printBoard()
{
	for (int i = 0; i < boardSize; ++i)
	{
		for (int j = 0; j < boardSize; ++j)
			std::cout << board[i][j] << ' ';
		std::cout << '\n';
	}
}

void Game_2048::setBoard(const boardType& board)
{
	this->board = board;
}

void Game_2048::resetBoard()
{
	for (int i = 0; i < boardSize; ++i)
		for (int j = 0; j < boardSize; ++j)
			board[i][j] = 0;
		
	addTwoTiles();
}

void Game_2048::addTwoTiles()
{
	for (int i = 0; i < 2; ++i)
		addRandomTile();
}


boardType Game_2048::getBoard() const
{
	return board;
}

std::vector<int> Game_2048::getAvailableMoves(const boardType& board,
											  const int& boardSize)
{
	std::set<int> availableMoves;
	bool stopSearching = false;
	for(int i = 0; i < boardSize && stopSearching == false; ++i)
		for (int j = 0; j < boardSize && stopSearching == false; ++j)
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

			if (availableMoves.size() == 4)
				stopSearching = true;
		}

	return std::vector<int>(availableMoves.begin(), availableMoves.end());
}

int Game_2048::sampleAction()
{
	std::vector<int> availableMoves(getAvailableMoves(board, boardSize));
	int action = -1;

	if (availableMoves.size() != 0)
	{
		std::uniform_int_distribution<int> indexesAvailable(0, availableMoves.size() - 1);

		int index = indexesAvailable(RNG);

		action = availableMoves[index];
	}

	return action;
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
		addRandomTile();
}

void Game_2048::addRandomTile()
{
	std::vector<std::pair<int, int> > emptyTiles;

	emptyTiles = getEmptyPositions();

	if (emptyTiles.size() != 0)
	{
		assignValueToRandomEmptyCell(emptyTiles);
	}

}

std::vector<std::pair<int, int> > Game_2048::getEmptyPositions()
{
	std::vector<std::pair<int, int> > emptyTiles;

	for (int i = 0; i != boardSize; ++i)
		for (int j = 0; j != boardSize; ++j)
			if (board[i][j] == 0)
				emptyTiles.push_back(std::make_pair(i, j));

	return emptyTiles;
}

void Game_2048::assignValueToRandomEmptyCell(std::vector<std::pair<int, int>>& emptyTiles)
{
	std::uniform_int_distribution<int> emptyCellDist(0, emptyTiles.size() - 1);

	std::pair<int, int> position = emptyTiles[emptyCellDist(RNG)];

	std::uniform_real_distribution<double> valueDistribution(0, 1.0);

	double probability = valueDistribution(RNG);

	board[position.first][position.second] = probability >= 0.9 ? 4 : 2;
}

void Game_2048::move(const int& yDirection, const int& xDirection)
{
	std::vector< std::vector<bool> > cellWasCombined(boardSize, std::vector<bool>(boardSize, false));

	int startX, endX, stopX, stepX;
	int startY, endY, stopY, stepY;

	std::tie(startX, endX, stopX, stepX) = getIterationElementsByDirection(xDirection);
	std::tie(startY, endY, stopY, stepY) = getIterationElementsByDirection(yDirection);

	for (int i = startY; i != endY; i += -stepY)
		for (int j = startX; j != endX; j += -stepX)
		{
			int line = i;
			int column = j;

			int newYPosition = line + yDirection;
			int newXPosition = column + xDirection;

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

fromToStopStep Game_2048::getIterationElementsByDirection(const int& direction)
{
	const int start = direction == 0 ? 0 : (direction == -1 ? 1 : boardSize - 1);
	const int end = direction == 0 ? boardSize : (direction == -1 ? boardSize : -1);
	const int stop = boardSize + end * direction;
	const int step = direction == 0 ? -1 : direction;

	return std::make_tuple(start, end, stop, step);
}

void Game_2048::setSeed(const int& seed)
{
	RNG.seed(seed);
}

void Game_2048::setFinishedIfNoActionIsAvailable()
{
	if(sampleAction() == -1)
		isGameFinished = true;
}

bool Game_2048::isFinished() const
{
	return isGameFinished;
}
