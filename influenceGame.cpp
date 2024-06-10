#include<armadillo>
#include "influenceGame.h"
#include<cstdlib>
#include<time.h>
#include<set>
#include<stack>
#include<queue>
#include<math.h>

influenceGame::influenceGame(int numActions, int numSeeds){
    n = numActions;
    k = numSeeds; 
}

// functions used to ensure network is connected
void generateEdges(arma::Mat<int> &graph, double p) {
    int cutoff = (int) 100.0*p;
    srand (time(NULL));
    int m = graph.n_rows;
    for (int i = 0; i < m; i++) {
        for (int j = i+1; j < m; j++){
            int draw = std::rand() % 100;
            if (draw < cutoff) {
                graph(i,j) = 1;
                graph(j,i) = 1;
            }
            else {
                graph(i,j) = 0;
                graph(j,i) = 0;
            }
        }
    }
}

// ensure network is connected
void makeConnected(arma::Mat<int> &graph) {
    std::set<int> visited; 
    std::stack<int> neighbors; 
    int m = graph.n_cols;

    srand (time(NULL));
    int startNode = rand() % m;

    neighbors.push(startNode);

    while (!neighbors.empty()) {  
        int cur = neighbors.top();
        neighbors.pop();

        visited.insert(cur);

        for (int j = 0; j < m; j++) {
            if (graph(cur, j) == 1 && visited.find(j) == visited.end()){
                neighbors.push(j);
            }
        }
    }

    std::vector<int> visitedVec(visited.begin(), visited.end());

    for (int i = 0; i < m; i++) {
        if (visited.find(i) == visited.end()) {
            int draw = rand() % visitedVec.size();
            graph(i, visitedVec[draw]) = 1;
            graph(visitedVec[draw], i) = 1;
            visitedVec.push_back(i);
        }
    }

}

// helper to verify if graph is now connected
bool verifyConnected(arma::Mat<int> &graph) {
    std::set<int> visited; 
    std::stack<int> neighbors; 

    neighbors.push(0);
    int m = graph.n_cols;

    while (!neighbors.empty()) {
        int cur = neighbors.top();
        neighbors.pop();

        visited.insert(cur);

        for (int j = 0; j < m; j++) {
            if (graph(cur, j) == 1 && visited.find(j) == visited.end()){
                neighbors.push(j);
            }
        }
    }

    if (visited.size() != m) {
        return false;
    } 

    return true;
}

void influenceGame::generateRandomNetwork(double prob) {
    network = arma::Mat<int>(n, n);
    generateEdges(network, prob);
    makeConnected(network);
}

void getDistance(int node, arma::Mat<int> graph, int* distances) {
    int m = graph.n_cols;

    std::queue<int> neighbors; 
    std::set<int> visited;

    neighbors.emplace(node);
    visited.insert(node);
    distances[node] = 0; 

    while (!neighbors.empty()) {
        int cur = neighbors.front();
        neighbors.pop();

        for (int j = 0; j < m; j++) {
            //std::cout << "looking at link between" << cur << " and " << j << std::endl;
            if (visited.find(j) == visited.end() && graph(cur, j) == 1) {
                neighbors.emplace(j);
                visited.insert(j);
                distances[j] = distances[cur] + 1;
            }
        }
    }
}

void influenceGame::computeDistances() {
    distanceMat = arma::Mat<int>(n,n);

    for (int row = 0; row < n; row++) {
        int rowDistances[n];
        
        getDistance(row, network, rowDistances); 

        /*
        std::cout << "Printing distances for row " << row << " "; 
        for (int i = 0; i < n; i++) {
            std::cout << rowDistances[i] << " ";
        }
        std::cout << std::endl;
        */

        for (int j = 0; j < n; j++) {
            distanceMat(row, j) = rowDistances[j];
        }
    }  
}

void influenceGame::constructPayoffMatrix() {
    payoffMat = arma::Mat<double>(n,n);

    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                payoffMat(i,j) = 0.5*n;
            }
            else {
                double payoff = 0.0;
                for (int k = 0; k < n; k++) {
                    if (distanceMat(i,k) < distanceMat(j,k)) {
                        payoff += 1.;
                    }
                    else if (distanceMat(i,k) == distanceMat(j,k)) {
                        payoff += .5;
                    }
                }
                payoffMat(i,j) = payoff;
                payoffMat(j,i) = n-payoff;
            }
        }
    }
}

void expWeights(arma::Mat<double> payoffs, arma::Mat<double> &rowPlayer, arma::Mat<double> &colPlayer, int numIters){
    // number of actions 
    int n = payoffs.n_rows;

    // define step size
    double eta = sqrt(log(n)/ (2*numIters));

    // define matrices to hold strategies for all T periods 
    arma::Mat<double> X(n,numIters);
    X.fill(1./n);
    //std::cout << "X mat " << X << std::endl;

    arma::Mat<double> Y(n,numIters);
    Y.fill(1./n);
    //std::cout << "Y mat " << Y << std::endl;

    for (int t = 1; t < numIters; t++) {
        //std::cout << "In period " << t << std::endl;
        // expected loss vector for row player 
        arma::Mat<double> lx = payoffs * Y.col(t-1);
        //std::cout << "Row loss " << lx << std::endl;

        // expected loss vector for col player 
        arma::Mat<double> ly = -1*payoffs.t() * X.col(t-1);
       //std::cout << "Col loss " << ly << std::endl;

        // update probabilites 
        arma::Mat<double> newx = X.col(t-1) % arma::exp(eta*lx);
        X.col(t) = newx * (1/arma::sum(newx));
        //std::cout << "Updated x prob " <<  X.col(t) << std::endl;

        arma::Mat<double> newy = Y.col(t-1) % arma::exp(eta*ly);
        Y.col(t) = newy * (1/arma::sum(newy));
        //std::cout << "Updated y prob " <<  Y.col(t) << std::endl;
    }

    rowPlayer = mean(X, 1);
    colPlayer = mean(Y, 1);
}


void influenceGame::findStaticEq() {
    int T = 500000;

    rowStrat = arma::Mat<double>(n,1); 
    colStrat = arma::Mat<double>(n,1);

    expWeights(payoffMat, rowStrat, colStrat, T);
} 

void influenceGame::computeCloseness() {
    arma::Mat<int> distanceSums = arma::sum(distanceMat, 1);
    arma::Mat<double> distanceSumsDbl(n,1);
    for (int i = 0; i < n; i++) {
        distanceSumsDbl(i,0) = (double) distanceSums(i,0);
    }
    closeness = (n-1)*pow(distanceSumsDbl, -1);
}

void influenceGame::computeDegree() {
    arma::Mat<int> degree = arma::sum(network,1);
    std::cout << degree << std::endl;
}