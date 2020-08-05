#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "..\Game_2048\Game_2048.h"


TEST_CASE("AvailableMoves-NoMovesAvailable","[Available Moves]")
{
	Game_2048 game{ 4 };

	std::vector<std::vector<int> > testBoard;
	std::vector<int> solution;

	testBoard = { {2,4,8,16},{4,8,16,32},{8,16,32,64},{16,32,64,128} };

	REQUIRE(Game_2048::getAvailableMoves(testBoard, 4) == solution);
}

TEST_CASE("AvailableMoves-VerticalMovesAvailable", "[Available Moves]")
{
	Game_2048 game{ 4 };

	std::vector<std::vector<int> > testBoard;
	std::vector<int> solution;

	testBoard = { {2,4,8,32},{4,8,16,32},{8,16,32,64},{16,32,64,128} };

	solution.push_back(Game_2048::Moves::UP   );
	solution.push_back(Game_2048::Moves::DOWN );
	
	REQUIRE(Game_2048::getAvailableMoves(testBoard, 4) == solution);
}

TEST_CASE("AvailableMoves-HorizontalMovesAvailable", "[Available Moves]")
{
	Game_2048 game{ 4 };

	std::vector<std::vector<int> > testBoard;
	std::vector<int> solution;

	testBoard = { {2,4,4,2},{4,8,16,32},{8,16,32,64},{16,32,64,128} };

	solution.push_back(Game_2048::Moves::RIGHT);
	solution.push_back(Game_2048::Moves::LEFT);

	REQUIRE(Game_2048::getAvailableMoves(testBoard, 4) == solution);
}

TEST_CASE("AvailableMoves-AllMovesAvailable", "[Available Moves]")
{
	Game_2048 game{ 4 };

	std::vector<std::vector<int> > testBoard;
	std::vector<int> solution;

	testBoard = { {2,4,8,8},{2,8,4,4},{2,4,32,4},{16,16,16,16} };

	solution.push_back(Game_2048::Moves::UP);
	solution.push_back(Game_2048::Moves::RIGHT);
	solution.push_back(Game_2048::Moves::DOWN);
	solution.push_back(Game_2048::Moves::LEFT);
	
	REQUIRE(Game_2048::getAvailableMoves(testBoard, 4) == solution);
}

TEST_CASE("GameIsFinished", "[FinishedIfNoActionIsAvailable]")
{
	boardType initializationBoard = { {2,4,8,16},{4,8,16,32},{8,16,32,64},{16,32,64,128} };
	Game_2048 game{ 4, initializationBoard };

	game.setFinishedIfNoActionIsAvailable();

	REQUIRE(game.isFinished() == true);
}

TEST_CASE("GameIsNotFinished", "[FinishedIfNoActionIsAvailable]")
{
	boardType initializationBoard = { {2,2,8,16},{4,8,16,32},{8,16,32,64},{16,32,64,128} };
	Game_2048 game{ 4, initializationBoard };

	game.setFinishedIfNoActionIsAvailable();

	REQUIRE(game.isFinished() == false);
}

TEST_CASE("ResetGame", "[Reset the game]")
{
	boardType initialBoard = { {2,2,8,16},{4,8,16,32},{8,16,32,64},{16,32,64,128} };

	Game_2048 game{ 4 , initialBoard };

	game.resetGame();

	boardType board = game.getBoard();

	int numberOf0Cells  = 0;
	int numberOf2Cells  = 0;
	int numberOf4Cells = 0;
	int numberOfOtherCells = 0;

	for(int row = 0; row < 4; ++row)
		for (int column = 0; column < 4; ++column)
		{
			switch (board[row][column])
			{
				case 0:
					numberOf0Cells += 1;
					break;

				case 2:
					numberOf2Cells += 1;
					break;

				case 4:
					numberOf4Cells += 1;
					break;

				default:
					numberOfOtherCells += 1;
					break;
			}
				
		}

	bool case_1 = numberOf2Cells == 2 && numberOf4Cells == 0 && numberOf0Cells == 14 && numberOfOtherCells == 0;
	bool case_2 = numberOf2Cells == 1 && numberOf4Cells == 1 && numberOf0Cells == 14 && numberOfOtherCells == 0;
	bool case_3 = numberOf2Cells == 0 && numberOf4Cells == 2 && numberOf0Cells == 14 && numberOfOtherCells == 0;

	REQUIRE(((case_1 == true) || (case_2 == true) || (case_3 == true)));

}

TEST_CASE("GetBoard", "[Get the board")
{
	boardType initializationBoard = { {2,2,8,16},{4,8,16,32},{8,16,32,64},{16,32,64,128} };
	Game_2048 game{ 4,  initializationBoard };

	REQUIRE(game.getBoard() == initializationBoard);
}

std::vector<boardType> getAllPossibleCombinations(const unsigned int& boardSize, boardType& board)
{
	std::vector<boardType> possibleCombinations;
	boardType copyOfBoard = board;

	for (int i = 0; i != boardSize; ++i)
		for (int j = 0; j != boardSize; ++j)
			if (board[i][j] == 0)
			{
				// Generate the two cell
				copyOfBoard[i][j] = 2;
				possibleCombinations.push_back(copyOfBoard);

				// Then the four one
				copyOfBoard[i][j] = 4;
				possibleCombinations.push_back(copyOfBoard);

				// Then reset to be the same as board
				copyOfBoard[i][j] = 0;
			}

	return possibleCombinations;
}

TEST_CASE("TakeAction-UP-NoCellsMove-NoMerge", "[Take Action]")
{
	boardType initialBoard = { {2,2,2,2},{4,4,4,4},{8,8,8,8},{16,16,16,16} };
	Game_2048 game{ 4, initialBoard};
	game.takeAction(Game_2048::Moves::UP);

	REQUIRE(game.getBoard() == initialBoard);
}

TEST_CASE("TakeAction-UP-CellsMove-NoMerge", "[Take Action]")
{
	boardType initialBoard = { {0,0,0,0},{4,4,4,4},{8,8,8,8},{16,16,16,16} };
	Game_2048 game{ 4, initialBoard };
	game.takeAction(Game_2048::Moves::UP);

	boardType boardAfterActionWithoutAddedCell = {{ 4,4,4,4 }, { 8,8,8,8 }, { 16,16,16,16 }, { 0,0,0,0 }};

	std::vector<boardType> possibleOutcomes = getAllPossibleCombinations(4, boardAfterActionWithoutAddedCell);

	auto position = std::find(possibleOutcomes.begin(), possibleOutcomes.end(), game.getBoard());

	REQUIRE(position != possibleOutcomes.end());
}

TEST_CASE("TakeAction-UP-CellsMove-CellsMerge", "[Take Action]")
{
	boardType initialBoard = { {4,4,4,4},{4,4,4,4},{4,4,4,4},{16,16,16,16} };
	Game_2048 game{ 4, initialBoard };

	game.takeAction(Game_2048::Moves::UP);

	boardType boardAfterActionWithoutAddedCell = { { 8,8,8,8 }, { 4,4,4,4 }, { 16,16,16,16 }, { 0,0,0,0 } };
	std::vector<boardType> possibleOutcomes = getAllPossibleCombinations(4, boardAfterActionWithoutAddedCell);

	auto position = std::find(possibleOutcomes.begin(), possibleOutcomes.end(), game.getBoard());

	REQUIRE(position != possibleOutcomes.end());
}

TEST_CASE("TakeAction-DOWN-NoCellsMove-NoMerge", "[Take Action]")
{
	boardType initialBoard = { {2,2,2,2},{4,4,4,4},{8,8,8,8},{16,16,16,16} };
	Game_2048 game{ 4, initialBoard };
	game.takeAction(Game_2048::Moves::DOWN);

	REQUIRE(game.getBoard() == initialBoard);
}

TEST_CASE("TakeAction-DOWN-CellsMove-NoMerge", "[Take Action]")
{
	boardType initialBoard = { {4,4,4,4},{8,8,8,8},{16,16,16,16},{0,0,0,0} };
	Game_2048 game{ 4, initialBoard };

	game.takeAction(Game_2048::Moves::DOWN);

	boardType boardAfterActionWithoutAddedCell = { { 0,0,0,0 }, { 4,4,4,4 }, { 8,8,8,8 }, { 16,16,16,16 } };

	std::vector<boardType> possibleOutcomes = getAllPossibleCombinations(4, boardAfterActionWithoutAddedCell);

	auto position = std::find(possibleOutcomes.begin(), possibleOutcomes.end(), game.getBoard());
	REQUIRE(position != possibleOutcomes.end());
}

TEST_CASE("TakeAction-DOWN-CellsMove-CellsMerge", "[Take Action]")
{
	boardType initialBoard = { {4,4,4,4},{4,4,4,4},{4,4,4,4},{16,16,16,16} };
	Game_2048 game{ 4, initialBoard };

	game.takeAction(Game_2048::Moves::DOWN);

	boardType boardAfterActionWithoutAddedCell = { { 0,0,0,0 }, { 4,4,4,4 }, { 8,8,8,8 }, { 16,16,16,16 }};
	std::vector<boardType> possibleOutcomes = getAllPossibleCombinations(4, boardAfterActionWithoutAddedCell);

	auto position = std::find(possibleOutcomes.begin(), possibleOutcomes.end(), game.getBoard());

	REQUIRE(position != possibleOutcomes.end());
}

TEST_CASE("TakeAction-LEFT-NoCellsMove-NoMerge", "[Take Action]")
{
	boardType initialBoard = { {2,4,2,4},{2,4,2,4},{8,16,8,16},{16,8,16,8} };
	Game_2048 game{ 4, initialBoard };
	game.takeAction(Game_2048::Moves::LEFT);

	REQUIRE(game.getBoard() == initialBoard);
}

TEST_CASE("TakeAction-LEFT-CellsMove-NoMerge", "[Take Action]")
{
	boardType initialBoard = { {0,2,4,8},{0,2,4,8},{0,2,4,8},{0,2,4,8} };
	Game_2048 game{ 4, initialBoard };
	game.takeAction(Game_2048::Moves::LEFT);

	boardType boardAfterActionWithoutAddedCell = { {2,4,8,0},{2,4,8,0},{2,4,8,0},{2,4,8,0} };

	std::vector<boardType> possibleOutcomes = getAllPossibleCombinations(4, boardAfterActionWithoutAddedCell);

	auto position = std::find(possibleOutcomes.begin(), possibleOutcomes.end(), game.getBoard());
	REQUIRE(position != possibleOutcomes.end());
}

TEST_CASE("TakeAction-LEFT-CellsMove-CellsMerge", "[Take Action]")
{
	boardType initialBoard = { {2,2,4,8},{2,2,4,8},{2,2,4,8},{2,2,4,8} };
	Game_2048 game{ 4, initialBoard };
	game.takeAction(Game_2048::Moves::LEFT);

	boardType boardAfterActionWithoutAddedCell = { {4,4,8,0},{4,4,8,0},{4,4,8,0},{4,4,8,0} };

	std::vector<boardType> possibleOutcomes = getAllPossibleCombinations(4, boardAfterActionWithoutAddedCell);

	auto position = std::find(possibleOutcomes.begin(), possibleOutcomes.end(), game.getBoard());
	REQUIRE(position != possibleOutcomes.end());
}


TEST_CASE("TakeAction-RIGHT-NoCellsMove-NoMerge", "[Take Action]")
{
	boardType initialBoard = { {2,4,2,4},{2,4,2,4},{8,16,8,16},{16,8,16,8} };
	Game_2048 game{ 4, initialBoard };
	game.takeAction(Game_2048::Moves::RIGHT);

	REQUIRE(game.getBoard() == initialBoard);
}

TEST_CASE("TakeAction-RIGHT-CellsMove-NoMerge", "[Take Action]")
{
	boardType initialBoard = { {2,4,8,0},{2,4,8,0},{2,4,8,0},{2,4,8,0} };
	Game_2048 game{ 4, initialBoard };
	game.takeAction(Game_2048::Moves::RIGHT);

	boardType boardAfterActionWithoutAddedCell = { {0,2,4,8},{0,2,4,8},{0,2,4,8},{0,2,4,8} };

	std::vector<boardType> possibleOutcomes = getAllPossibleCombinations(4, boardAfterActionWithoutAddedCell);

	auto position = std::find(possibleOutcomes.begin(), possibleOutcomes.end(), game.getBoard());
	REQUIRE(position != possibleOutcomes.end());
}

TEST_CASE("TakeAction-RIGHT-CellsMove-CellsMerge", "[Take Action]")
{
	boardType initialBoard = { {4,4,8,0},{4,4,8,0},{4,4,8,0},{4,4,8,0} };
	Game_2048 game{ 4, initialBoard };
	game.takeAction(Game_2048::Moves::RIGHT);

	boardType boardAfterActionWithoutAddedCell = { {0,0,8,8},{0,0,8,8},{0,0,8,8},{0,0,8,8} };

	std::vector<boardType> possibleOutcomes = getAllPossibleCombinations(4, boardAfterActionWithoutAddedCell);

	auto position = std::find(possibleOutcomes.begin(), possibleOutcomes.end(), game.getBoard());
	REQUIRE(position != possibleOutcomes.end());
}