#include <iostream>
#include "NeuralNetwork.h"
#include <vector>
#include <cstdio>

int main()
{
	std::vector<uint32_t> topology = { 2,3,1 };
	sp::SimpleNeuralNetwork nn(topology, 0.1);

	std::vector<std::vector<float>> targetInputs = {
		{0.0f,0.0f},
		{1.0f,1.0f},
		{1.0f,0.0f},
		{0.0f,1.0f}
	};
	std::vector<std::vector<float>> targetOutputs = {
		{0.0f},
		{0.0f},
		{1.0f},
		{1.0f}
	};

	uint32_t epoch = 10000;


	std::cout << "training started\n";
	for (uint32_t i = 0; i < epoch; i++)
	{
		uint32_t index = rand() % 4;
		nn.FeedForward(targetInputs[index]);
		nn.backPropogate(targetOutputs[index]);
	}

	std::cout << "training completed\n";

	for (std::vector<float> input : targetInputs)
	{
		nn.FeedForward(input);
		std::vector<float> preds = nn.getPrediction();
		std::cout << input[0] << "," << input[1] << " => " << preds[0] << std::endl;
	}

	return 0;
}




