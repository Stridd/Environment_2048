#include "Game_2048.h"
#include <chrono>
#include <iostream>

using namespace std::chrono;

int main()
{
	Game_2048 game{ 4 };
	auto start = high_resolution_clock::now();
	for (int i = 0; i < 1000; ++i)
	{
		std::cout << i << '\n';
		game.resetGame();

		while (!game.isFinished())
		{
			game.takeAction(game.sampleAction());
		}
	}
	auto end = high_resolution_clock::now();

	auto duration = duration_cast<milliseconds>(end - start);

	std::cout << duration.count();

	return 0;
}