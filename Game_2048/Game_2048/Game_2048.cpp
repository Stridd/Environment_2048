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

Game_2048::Game_2048(const unsigned int& boardSize, boardType& board)
{
	this->boardSize = boardSize;
	this->board = board;
	currentEpisode = 0;
	isGameFinished = false;

	// We want to set the game as finished based on the input board
	setFinishedIfNoActionIsAvailable();

	RNG.seed(std::random_device()());
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

void Game_2048::resetBoard()
{
	for (int i = 0; i < boardSize; ++i)
		for (int j = 0; j < boardSize; ++j)
			board[i][j] = 0;
		
	addTwoTiles();
}

bool Game_2048::canBeMergedAtPositions(const int& row, 
									   const int& column, 
									   const int& afterMoveRow, 
									   const int& afterMoveColumn,
									   std::vector< std::vector<bool> >& cellWasCombined)
{
	return board[row][column] == board[afterMoveRow][afterMoveColumn] &&
		!cellWasCombined[row][column] &&
		!cellWasCombined[afterMoveRow][afterMoveColumn];
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

// Used for the exploration part of the agent.
// We want to make sure the agent is able to take only the available actions
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
		emptyMergedCellsInformation();
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
	emptyPositionsVector emptyTiles;

	emptyTiles = getEmptyPositions();

	if (emptyTiles.size() != 0)
		assignValueToRandomEmptyCell(emptyTiles);

}

emptyPositionsVector Game_2048::getEmptyPositions()
{
	emptyPositionsVector emptyTiles;

	for (int i = 0; i != boardSize; ++i)
		for (int j = 0; j != boardSize; ++j)
			if (board[i][j] == 0)
				emptyTiles.push_back(std::make_pair(i, j));

	return emptyTiles;
}

void Game_2048::assignValueToRandomEmptyCell(emptyPositionsVector& emptyTiles)
{
	std::uniform_int_distribution<int> emptyCellDist(0, emptyTiles.size() - 1);

	std::pair<int, int> randomPosition = emptyTiles[emptyCellDist(RNG)];

	std::uniform_real_distribution<double> valueDistribution(0, 1.0);

	double cellValueProbability = valueDistribution(RNG);

	board[randomPosition.first][randomPosition.second] = cellValueProbability >= 0.9 ? 4 : 2;
}

void Game_2048::move(const int& yDirection, const int& xDirection)
{
	std::vector< std::vector<bool> > cellWasCombined(boardSize, std::vector<bool>(boardSize, false));

	int startColumn, endColumn, stopColumn, stepColumn;
	int startRow, endRow, stopRow, stepRow;

	// In order to avoid creating four functions (MoveUp, MoveDown, MoveRight, MoveLeft) 
	// I added some function that gives the boundaries in form of start-end-stop-step
	std::tie(startColumn, endColumn, stopColumn, stepColumn) = getIterationElementsByDirection(xDirection);
	std::tie(startRow, endRow, stopRow, stepRow)             = getIterationElementsByDirection(yDirection);

	for (int y = startRow; y != endRow; y += -stepRow)
		for (int x = startColumn; x != endColumn; x += -stepColumn)
		{
			int row = y;
			int column = x;

			int afterMoveRow    = row + yDirection;
			int afterMoveColumn = column + xDirection;
	
			while (row != stopRow && column != stopColumn && board[row][column] != 0)
			{
				if(canBeMergedAtPositions(row,column,afterMoveRow,afterMoveColumn,cellWasCombined))
				{
					storeMergedCellsInformation(board[afterMoveRow][afterMoveColumn]);

					board[afterMoveRow][afterMoveColumn] *= 2;
					board[row][column] = 0;

					cellWasCombined[afterMoveRow][afterMoveColumn] = true;
					
				}
				else
					if (board[afterMoveRow][afterMoveColumn] == 0)
					{
						board[afterMoveRow][afterMoveColumn] = board[row][column];
						board[row][column] = 0;
					}

				row          += yDirection;
				afterMoveRow += yDirection;

				column          += xDirection;
				afterMoveColumn += xDirection;
			}
		}
}

void Game_2048::storeMergedCellsInformation(int& value)
{
	if (mergedCellsAfterMove.find(value) != mergedCellsAfterMove.end())
	{
		mergedCellsAfterMove[value] += 2;
	}
	else
		mergedCellsAfterMove[value] = 2;
}


fromToStopStep Game_2048::getIterationElementsByDirection(const int& direction)
{
	const int startAt = direction == 0 ? 0 : (direction == -1 ? 1 : boardSize - 1);
	const int endAt = direction == 0 ? boardSize : (direction == -1 ? boardSize : -1);
	const int stopAt = boardSize + endAt * direction;
	const int step = direction == 0 ? -1 : direction;

	return std::make_tuple(startAt, endAt, stopAt, step);
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

std::unordered_map<int, int> Game_2048::getMergedCellsAfterMove()
{
	return mergedCellsAfterMove;
}

void Game_2048::emptyMergedCellsInformation()
{
	mergedCellsAfterMove.clear();
}
