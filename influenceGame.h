#ifndef INFLUENCE_GAME_H
#define INFLUENCE_GAME_H

#include<armadillo>
#include<cstdlib>
#include<cstdio>

class influenceGame 
{
    private: 
        // number of nodes in network
        int n;
        // number of seeds in game 
        int k; 

    public:
        // matrix to hold payoffs 
        arma::Mat<double> payoffMat;

        // vectors to hold strategies
        arma::Mat<double> rowStrat;
        arma::Mat<double> colStrat;

        // matrix to hold network
        arma::Mat<int> network;

        // matrix to hold distances
        arma::Mat<int> distanceMat;

        // matri to hold closeness
        arma::Mat<double> closeness;

        // constructors
        influenceGame() = default;
        influenceGame(int numActions, int numSeeds); 

        // destructor 
        ~influenceGame()= default;

        // function to generate randomNetwork
        void generateRandomNetwork(double prob);

        // function to find distances 
        void computeDistances();

        // function to check closeness centrality
        void computeCloseness();

        // function to check degree centrality
        void computeDegree();

        // function to create payoff matrix 
        void constructPayoffMatrix();

        // function to find equilibrium 
        void findStaticEq();
};

#endif