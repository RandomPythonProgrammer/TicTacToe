#include "Table.h"
#include "Board.h"
#include "Network.h"
#include <torch/torch.h>
#include <execution>
#include <toml.hpp>
#include <ranges>
#include <numeric>

int main() {
    const toml::value config = toml::parse("config.toml");
    Table table(toml::find<std::string>(config, "database"));
    int numBoards = toml::find<int>(config, "boards");
    float epsilon = toml::find<float>(config, "epsilon"); 
    int batchSize = toml::find<int>(config, "batch_size");
    Network network;
    
    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) {
        std::cerr << "Failed to start CUDA, using CPU" << std::endl;
        device = torch::Device(torch::kCPU);
    }

    network.to(device);
    torch::nn::MSELoss lossFunction;
    torch::optim::Adam optimizer(network.parameters());

    //run simulation
    int epochs = toml::find<int>(config, "epochs");
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::vector<Board> boards = std::vector<Board>(numBoards);
        std::atomic<bool> complete = false;
        std::vector<std::vector<torch::Tensor>> states(numBoards);
        std::cout << "Running epoch: " + std::to_string(epoch) << std::endl;
        while (!complete) {
            complete = true;
            std::vector<std::vector<Move>> moves(numBoards);
            std::vector<int> sizes(numBoards);
            std::atomic<int> num_moves = 0;

            std::vector<int> range(numBoards);
            std::iota(range.begin(), range.end(), 0);

            std::for_each(std::execution::par, range.begin(), range.end(), [&](int& index){
                Board& board = boards[index];
                if (board.getResult() == Result::NONE) {
                    moves[index] = board.getMoves();
                    sizes[index] = moves[index].size();
                    num_moves += moves[index].size();
                    complete = false;
                }
            });

            torch::Tensor evaluationBuffer = torch::zeros({num_moves, 3, 3, 3});
            std::vector<int> offsets(numBoards + 1, 0);
            std::partial_sum(sizes.begin(), sizes.end(), offsets.begin()+1);

            std::for_each(std::execution::par, range.begin(), range.end(), [&](int& index){
                Board& board = boards[index];
                if (board.getResult() == Result::NONE) {
                    int offset =  offsets[index];
                    for (int i = 0; i < sizes[index]; i++) {
                        Move& move = moves[index][i];
                        board.makeMove(move);
                        evaluationBuffer[offset+i] = board.getData();
                        board.undo();
                    }
                }
            });

            evaluationBuffer = evaluationBuffer.to(device);
            torch::Tensor evaluation = network.forward(evaluationBuffer);

            std::for_each(std::execution::par, range.begin(), range.end(), [&](int& index){
                Board& board = boards[index];
                if (board.getResult() == Result::NONE) {
                    int offset = offsets[index];
                    int top = torch::rand({1}).item<float>() * sizes[index];
                    if (torch::rand({1}).item<float>() > epsilon) {
                        torch::Tensor scores = evaluation.slice(0, offset, offset+sizes[index]);
                        if (board.getTurn() == Player::cross) {
                            top = torch::argmax(scores).item<int>();
                        } else {
                            top = torch::argmin(scores).item<int>();
                        }
                    }
                    states[index].push_back(board.getData());
                    board.makeMove(moves[index][top]);
                }
            });
        }

        //update the q values
        for (int i = 0; i < numBoards; i++) {
            float score = 0;
            Result outcome = boards[i].getResult();
            if (outcome == Result::CROSS) {
                score = 1;
            } else if (outcome == Result::CIRCLE) {
                score = -1;
            }

            for (torch::Tensor& state: states[i]) {
                table.updateQ(state, score);
            }
        }

        //train using a partial dataset
        std::pair<torch::Tensor, torch::Tensor> trainingData = table.getDataset(batchSize);
        torch::Tensor x = trainingData.first.to(device);
        torch::Tensor y = trainingData.second.view({-1, 1}).to(device);

        optimizer.zero_grad();
        torch::Tensor output = network.forward(x);
        torch::Tensor loss = lossFunction->forward(output, y);
        loss.backward();

        std::cout << "Average Loss: " << loss.mean() << std::endl;
        optimizer.step();
    }
}