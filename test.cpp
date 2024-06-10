#include<cstdio>
#include "influenceGame.h"

int main(int argc, char* argv[]) {
    // testing constructor and network declaration
    std::cout << "Testing constructor and random network generation" << std::endl;
    int networkSize = std::stoi(argv[1]);
    influenceGame testNet(networkSize, 1);
    double edgeProb = std::stod(argv[2]);
    testNet.generateRandomNetwork(edgeProb);
    std::cout << testNet.network << std::endl;

    //testing distance computation 
    //std::cout << "Testing distance computation" << std::endl;
    testNet.computeDistances();
    std::cout << testNet.distanceMat << std::endl;

    // test payoff matrix 
    //std::cout << "Testing payoff matrix computation" << std::endl;
    testNet.constructPayoffMatrix();
    std::cout << testNet.payoffMat << std::endl;

    std::cout << "Testing static equilibrium computation" << std::endl;
    testNet.findStaticEq();
    std::cout << "Row player strategy" << testNet.rowStrat << std::endl;
    std::cout << "Column player strategy" << testNet.colStrat << std::endl;

    arma::Mat<double> rowPayoff = testNet.rowStrat.t() * testNet.payoffMat * testNet.colStrat;
    arma::Mat<double> colPayoff = testNet.colStrat.t() * -testNet.payoffMat.t() * testNet.rowStrat;

    std::cout << "Row player payoff" <<  rowPayoff << std::endl;
    std::cout << "Col player payoff" <<  colPayoff << std::endl;

    testNet.computeCloseness();
    std::cout << "Closeness centrality of all nodes: " << testNet.closeness << std::endl;

    std::cout << "Degree of all nodes: " << std::endl;
    testNet.computeDegree();

    return 0;
}