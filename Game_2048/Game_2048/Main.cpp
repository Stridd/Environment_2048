#include "Game_2048.h"
#include <chrono>
#include <iostream>
#include <array>
#include <numeric>

using namespace std::chrono;

int main()
{
	int trials		= 10;
	int trialSize	= 100000;

	std::vector<long long> trialDuration(trials, 0);

	for (int trialNumber = 0; trialNumber != trials; ++trialNumber)
	{
		auto start = high_resolution_clock::now();

		for (int trialIteration = 0; trialIteration != trialSize; ++trialIteration)
		{
			Game_2048 game{ 4 };

			game.resetGame();

			while (!game.isFinished())
			{
				// Split the sample action and the action itself to increase speed
				int action = game.sampleAction();
				if (action == -1)
					// This leaves the responsibility to the one who uses it to set the game to finished
					game.setFinishedIfNoActionIsAvailable();
				else
					game.takeAction(action);
			}

		}

		auto end	  = high_resolution_clock::now();
		auto duration = duration_cast<seconds>(end - start);

		trialDuration[trialNumber] = duration.count();
		std::cout << "Trial: " << trialNumber << " took " << trialDuration[trialNumber] << " seconds " << '\n';
	}

	double meanDuration = std::accumulate(trialDuration.begin(), trialDuration.end(), 0.0) / trials;

	std::cout << "Mean duration of experiments: " << meanDuration;

	return 0;
}