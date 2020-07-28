#include "Game_2048.h"
#include <chrono>
#include <iostream>

using namespace std::chrono;

int main()
{
	Game_2048 game{ 4 };
	game.setSeed(0);
	auto start = high_resolution_clock::now();
	for (int i = 0; i < 10000; ++i)
	{
		game.resetGame();
		std::cout << i << '\n';
		while (!game.isFinished())
		{
			// Split the sample action and the action itself to increase speed
			int action = game.sampleAction();
			if (action == -1)
				// This leaves the responsibility to the one who uses it to set the game to finished
				game.setFinished();
			else
				game.takeAction(action);
		}
	}
	auto end = high_resolution_clock::now();

	auto duration = duration_cast<seconds>(end - start);

	std::cout << duration.count();

	return 0;
}