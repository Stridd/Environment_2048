#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "..\Game_2048\Game_2048.h"

TEST_CASE("AVAILABLE MOVES")
{
	Game_2048 game{ 4 };

	std::vector<std::vector<int> > testBoard = { {4, 4, 4, 4},{2,2,2,2},{8,8,8,8},{16,16,16,16} };
	std::vector<int> solution = { Game_2048::Moves::RIGHT, Game_2048::Moves::LEFT };

	SECTION("TWO MOVES AVAILABLE")
	{
		REQUIRE(game.getAvailableMoves(testBoard, 4) == solution);
	}
	

}