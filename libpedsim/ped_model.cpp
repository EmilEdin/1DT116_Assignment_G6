//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>

// This checks if we are on ARM (Mac M1/M2) or Intel (Lab computers)
#if defined(__ARM_NEON) || defined(__aarch64__)
#include "sse2neon.h" // Translates Intel intrinsics to ARM NEON
#else
#include <immintrin.h> // Standard Intel SSE/AVX for lab machines
#endif

#include <cmath>
#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>

// Assignment 2
// #include <emmintrin.h> // SSE2
// #include <smmintrin.h> // SSE4.1

void Ped::Model::setup(std::vector<Ped::Tagent *> agentsInScenario, std::vector<Twaypoint *> destinationsInScenario, IMPLEMENTATION implementation)
{
#ifndef NOCUDA
	// Convenience test: does CUDA work on this machine?
	cuda_test();
#else
	std::cout << "Not compiled for CUDA" << std::endl;
#endif

	// Set
	agents = std::vector<Ped::Tagent *>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint *>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();

	// Assignment 2
	// --- ADD THIS BLOCK BELOW OR YOU DIE ---

	// 1. Allocate Memory: Make our new arrays the same size as the number of agents
	int N = agents.size();
	agentX.resize(N + 4);
	agentY.resize(N + 4);
	destX.resize(N + 4);
	destY.resize(N + 4);
	destR.resize(N + 4);
	// 2. Populate with initial data
	for (int i = 0; i < N; ++i)
	{
		agentX[i] = agents[i]->getX();
		agentY[i] = agents[i]->getY();

		// Force agent to pick its first waypoint
		agents[i]->computeNextDesiredPosition();

		// Implementation 1: Save that waypoint into our SoA arrays
		if (agents[i]->hasDestination())
		{
			destX[i] = agents[i]->getDestX();
			destY[i] = agents[i]->getDestY();
			destR[i] = agents[i]->getDestR();
		}
		else
		{
			// If no destination, just target current position (stay still)
			destX[i] = agentX[i];
			destY[i] = agentY[i];
			destR[i] = 0;
		}
		/*
			// Implementation 2:
			for (int i = 0; i < agents.size(); ++i)
		{
			// Copy current position
			agentX[i] = agents[i]->getX();
			agentY[i] = agents[i]->getY();

			// Ask the agent to calculate its first target so we have a valid destination
			agents[i]->computeNextDesiredPosition();

			// Copy that destination into our arrays
			destX[i] = agents[i]->getDesiredX();
			destY[i] = agents[i]->getDesiredY();
		}
		*/
	}
}

void Ped::Model::tick()
{
	switch (this->implementation)
	{
	case Ped::VECTOR:
	{
		int N = agents.size();

		// --- LOOP 1: SIMD MOVEMENT (The Fast Part) ---
		// We step by 4 because SSE processes 4 floats at a time
		for (int i = 0; i < N; i += 4)
		{
			// 1. Load data from arrays into registers
			// (Note: Assuming you changed vectors to float for easy SIMD math)
			__m128 curX = _mm_loadu_ps(&agentX[i]);
			__m128 curY = _mm_loadu_ps(&agentY[i]);
			__m128 dstX = _mm_loadu_ps(&destX[i]);
			__m128 dstY = _mm_loadu_ps(&destY[i]);

			// 2. Calculate Direction (Target - Current)
			__m128 diffX = _mm_sub_ps(dstX, curX);
			__m128 diffY = _mm_sub_ps(dstY, curY);

			// 3. Calculate Length (sqrt(diffX^2 + diffY^2))
			__m128 sqX = _mm_mul_ps(diffX, diffX);
			__m128 sqY = _mm_mul_ps(diffY, diffY);
			__m128 sum = _mm_add_ps(sqX, sqY);
			__m128 len = _mm_sqrt_ps(sum);

			// 4. Normalize and Move (NextPos = Cur + (Diff / Len))
			// We add a tiny epsilon (0.001) to Len to avoid division by zero if len is 0
			__m128 epsilon = _mm_set1_ps(0.001f);
			len = _mm_max_ps(len, epsilon);

			__m128 stepX = _mm_div_ps(diffX, len);
			__m128 stepY = _mm_div_ps(diffY, len);

			// Round to nearest integer (if your logic requires int steps)
			// or just add if keeping as floats. Let's add:
			__m128 nextX = _mm_add_ps(curX, stepX);
			__m128 nextY = _mm_add_ps(curY, stepY);

			// 5. Store back to arrays
			_mm_storeu_ps(&agentX[i], nextX);
			_mm_storeu_ps(&agentY[i], nextY);
		}

		// --- LOOP 2: CLEANUP & LOGIC (The Slow Part) ---
		for (int i = 0; i < N; ++i)
		{
			// 1. Sync the Object (Required so the GUI sees the movement!)
			agents[i]->setX((int)agentX[i]);
			agents[i]->setY((int)agentY[i]);

			// 2. Check if reached destination
			// (We can use the arrays for this check to be fast)
			float dx = destX[i] - agentX[i];
			float dy = destY[i] - agentY[i];
			float distSq = dx * dx + dy * dy;
			float radiusSq = destR[i] * destR[i];

			if (distSq < radiusSq)
			{
				// We arrived! Use the Object to get the NEXT waypoint.
				agents[i]->computeNextDesiredPosition();

				// Update our fast arrays with the new target
				if (agents[i]->hasDestination())
				{
					destX[i] = agents[i]->getDestX();
					destY[i] = agents[i]->getDestY();
					destR[i] = agents[i]->getDestR();
				}
			}
		}
		break;
	}

	case Ped::SEQ:
		for (int i = 0; i < agents.size(); i++)
		{
			agents[i]->computeNextDesiredPosition();
			agents[i]->setX(agents[i]->getDesiredX());
			agents[i]->setY(agents[i]->getDesiredY());
		}
		break;

	case Ped::OMP:
	{
#pragma omp parallel for
		for (int i = 0; i < agents.size(); i++)
		{
			agents[i]->computeNextDesiredPosition();
			agents[i]->setX(agents[i]->getDesiredX());
			agents[i]->setY(agents[i]->getDesiredY());
		}
		break;
	}
	case Ped::PTHREAD:
	{
		int numThreads = std::thread::hardware_concurrency();
		if (numThreads == 0)
			numThreads = 2;

		std::vector<std::thread> threads;
		int n = agents.size();
		int chunkSize = (n + numThreads - 1) / numThreads;

		for (int t = 0; t < numThreads; t++)
		{
			int start = t * chunkSize;
			if (start >= n)
				break;

			int end = std::min(start + chunkSize, n);

			threads.emplace_back([this, start, end]()
								 {
					for (int i = start; i < end; ++i) {
                		agents[i]->computeNextDesiredPosition();
                		agents[i]->setX(agents[i]->getDesiredX());
                		agents[i]->setY(agents[i]->getDesiredY());
            		} });
		}
		for (auto &th : threads)
		{
			if (th.joinable())
			{
				th.join();
			}
		}

		break;
	}
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int>> takenPositions;
	for (std::set<const Ped::Tagent *>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt)
	{
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int>> prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else
	{
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int>>::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it)
	{

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end())
		{

			// Set the agent's position
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent *> Ped::Model::getNeighbors(int x, int y, int dist) const
{

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)
	return set<const Ped::Tagent *>(agents.begin(), agents.end());
}

void Ped::Model::cleanup()
{
	// Nothing to do here right now.
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent)
				  { delete agent; });
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination)
				  { delete destination; });
}
