#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "..\Game_2048\Game_2048.h"


TEST_CASE("AvailableMovesNoMovesAvailable","[Available Moves]")
{
	Game_2048 game{ 4 };

	std::vector<std::vector<int> > testBoard;
	std::vector<int> solution;

	testBoard = { {2,4,8,16},{4,8,16,32},{8,16,32,64},{16,32,64,128} };

	REQUIRE(Game_2048::getAvailableMoves(testBoard, 4) == solution);
}

TEST_CASE("AvailableMovesVerticalMovesAvailable", "[Available Moves]")
{
	Game_2048 game{ 4 };

	std::vector<std::vector<int> > testBoard;
	std::vector<int> solution;

	testBoard = { {2,4,8,32},{4,8,16,32},{8,16,32,64},{16,32,64,128} };

	solution.push_back(Game_2048::Moves::UP   );
	solution.push_back(Game_2048::Moves::DOWN );
	
	REQUIRE(Game_2048::getAvailableMoves(testBoard, 4) == solution);
}

TEST_CASE("AvailableMovesHorizontalMovesAvailable", "[Available Moves]")
{
	Game_2048 game{ 4 };

	std::vector<std::vector<int> > testBoard;
	std::vector<int> solution;

	testBoard = { {2,4,4,2},{4,8,16,32},{8,16,32,64},{16,32,64,128} };

	solution.push_back(Game_2048::Moves::RIGHT);
	solution.push_back(Game_2048::Moves::LEFT);

	REQUIRE(Game_2048::getAvailableMoves(testBoard, 4) == solution);
}

TEST_CASE("AvailableMovesAllMovesAvailable", "[Available Moves]")
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
	Game_2048 game{ 4 };	
	game.setBoard({ {2,4,8,16},{4,8,16,32},{8,16,32,64},{16,32,64,128} });

	game.setFinishedIfNoActionIsAvailable();

	REQUIRE(game.isFinished() == true);
}

TEST_CASE("GameIsNotFinished", "[FinishedIfNoActionIsAvailable]")
{
	Game_2048 game{ 4 };
	game.setBoard({ {2,2,8,16},{4,8,16,32},{8,16,32,64},{16,32,64,128} });

	game.setFinishedIfNoActionIsAvailable();

	REQUIRE(game.isFinished() == false);
}

TEST_CASE("ResetGame", "[Reset the game]")
{
	Game_2048 game{ 4 };
	game.setBoard({ {2,2,8,16},{4,8,16,32},{8,16,32,64},{16,32,64,128} });

	game.resetGame();

	boardType board = game.getBoard();

	int nr_0  = 0;
	int nr_2  = 0;
	int nr_4  = 0;
	int other = 0;

	for(int row = 0; row < 4; ++row)
		for (int column = 0; column < 4; ++column)
		{
			switch (board[row][column])
			{
				case 0:
					nr_0 += 1;
					break;

				case 2:
					nr_2 += 1;
					break;

				case 4:
					nr_4 += 1;
					break;

				default:
					other += 1;
					break;
			}
				
		}

	bool case_1 = nr_2 == 2 && nr_4 == 0 && nr_0 == 14 && other == 0;
	bool case_2 = nr_2 == 1 && nr_4 == 1 && nr_0 == 14 && other == 0;
	bool case_3 = nr_2 == 0 && nr_4 == 2 && nr_0 == 14 && other == 0;

	REQUIRE(((case_1 == true) || (case_2 == true) || (case_3 == true)));

}