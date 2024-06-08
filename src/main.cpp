#include "Table.h"
#include "Board.h"
#include "ResNetImpl.h"
#include <torch/torch.h>
#include <toml.hpp>
#include <ranges>
#include <numeric>
#include <omp.h>

void train() {
    const toml::value config = toml::parse("config.toml");
    Table table(toml::find<std::string>(config, "database"));
    int numBoards = toml::find<int>(config, "boards");
    float epsilon = toml::find<float>(config, "epsilon"); 
    int batchSize = toml::find<int>(config, "batch_size");
    std::string savePath = toml::find<std::string>(config, "save_path");
    omp_set_num_threads(omp_get_max_threads());

    ResNet network;
    
    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) {
        std::cerr << "Failed to start CUDA, using CPU" << std::endl;
        device = torch::Device(torch::kCPU);
    }

    network->to(device);
    torch::nn::MSELoss lossFunction;
    torch::optim::Adam optimizer(network->parameters());

    //run simulation
    int epochs = toml::find<int>(config, "epochs");
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch: " << epoch << std::endl;
        std::vector<Board> boards = std::vector<Board>(numBoards);
        std::atomic<bool> complete = false;
        std::vector<std::vector<torch::Tensor>> states(numBoards);
        std::vector<double> scores(numBoards, 0);
        int turn = 0;
        std::cout << "Running Simulation" << std::endl;
        while (!complete) {
            complete = true;
            std::cout << "Turn: " << turn;
            std::vector<std::vector<Move>> moves(numBoards);
            std::vector<int> sizes(numBoards);
            std::atomic<int> num_moves = 0;

            std::vector<int> range(numBoards);
            std::iota(range.begin(), range.end(), 0);

            #pragma omp parallel for
            for (int& index: range) {
                Board& board = boards[index];
                if (board.getResult() == Result::NONE) {
                    moves[index] = board.getMoves();
                    sizes[index] = moves[index].size();
                    num_moves += moves[index].size();
                    complete = false;
                }
            }

            torch::Tensor evaluationBuffer = torch::zeros({num_moves, 3, 3, 3});
            std::vector<int> offsets(numBoards + 1, 0);
            std::partial_sum(sizes.begin(), sizes.end(), offsets.begin()+1);

            #pragma omp parallel for
            for (int& index: range) {
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
            }

            evaluationBuffer = evaluationBuffer.to(device);
            torch::Tensor evaluation = network->forward(evaluationBuffer);

            #pragma omp parallel for
            for (int& index: range) {
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
                    states[index].push_back(board.getData().clone());
                    board.makeMove(moves[index][top]);

                    Result outcome = board.getResult();
                    if (outcome != Result::NONE) {
                        if (outcome == Result::CROSS) {
                            scores[index] = 1;
                        } else if (outcome == Result::CIRCLE) {
                            scores[index] = -1;
                        }
                    }
                }
            }
            turn++;
            std::cout << "\33[2K\r";
        }

        //update the q values
        std::cout << "Writing to Database" << std::endl;
        table.updateQ(states, scores);

        //train using a partial dataset
        std::cout << "Training" << std::endl;
        std::pair<torch::Tensor, torch::Tensor> trainingData = table.getDataset(batchSize);
        torch::Tensor x = trainingData.first.to(device);
        torch::Tensor y = trainingData.second.view({-1, 1}).to(device);

        optimizer.zero_grad();
        torch::Tensor output = network->forward(x);
        torch::Tensor loss = lossFunction->forward(output, y);
        loss.backward();

        std::cout << "Average Loss: " << loss.mean() << std::endl;
        optimizer.step();
    }

    torch::save(network, savePath);
}

void test() {
    const toml::value config = toml::parse("config.toml");
    std::string savePath = toml::find<std::string>(config, "save_path");
    Table table(toml::find<std::string>(config, "database"));

    std::cout << "0: for cross (first)" << std::endl;
    std::cout << "1: for circle (second)" <<std::endl;
    std::cout << "Select: ";
    int option;
    std::cin >> option;
    Player human = (Player) option;

    Board board;
    ResNet network;
    torch::load(network, savePath);
    
    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) {
        std::cerr << "Failed to start CUDA, using CPU" << std::endl;
        device = torch::Device(torch::kCPU);
    }

    network->to(device);

    while (board.getResult() == Result::NONE) {
        torch::Tensor data = board.getData();
        if (board.getTurn() == human) {
            //displays the board on the screen
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if ((data[0][i][j] == 1).item<bool>()) {
                        std::cout << "X";
                    } else if ((data[1][i][j] == 1).item<bool>()) {
                        std::cout << "O";
                    } else {
                        std::cout << " ";
                    }
                    std::cout << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "Table Value: " << table.getQ(data) << std::endl;
            Move move;
            std::cout << "Row: ";
            std::cin >> move.row;
            std::cout << "Col: ";
            std::cin >> move.col;
            move.player = human;
            board.makeMove(move);
        } else {
            std::vector<Move> moves = board.getMoves();
            torch::Tensor buffer = torch::zeros({(long) moves.size(), 3, 3, 3});
            for (int i = 0; i < moves.size(); i++) {
                board.makeMove(moves[i]);
                buffer[i] = board.getData();
                board.undo();
            }
            buffer = buffer.to(device);
            torch::Tensor output = network->forward(buffer);
            int best;
            if (human == Player::circle) {
                best = torch::argmax(output).item<int>();
            } else {
                best = torch::argmin(output).item<int>();
            }
            board.makeMove(moves[best]);
        }
    }

}

int main() {
    std::cout << "0: for train" << std::endl;
    std::cout << "1: for test" <<std::endl;
    std::cout << "Select: ";
    int option;
    std::cin >> option;
    if (option == 0) {
        train();
    } else if (option == 1) {
        test();
    } else {
        std::cerr << "invalid option" << std::endl;
    }

    return 0;
}