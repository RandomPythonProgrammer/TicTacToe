#include "Board.h"

Board::Board() {
    data = torch::zeros({3, 3, 3});
    turn = Player::cross;
}

Board& Board::makeMove(Move move) {
    data[(int) move.player][move.row][move.col] = 1;
    turn = (Player) (1 - (int) turn);
    data[2].fill_((int) turn);
    moves.push(move);
    return *this;
}

Board& Board::undo() {
    Move move = moves.top();
    data[(int) move.player][move.row][move.col] = 0;
    turn = (Player) (1 - (int) turn);
    data[2].fill_((int) turn);
    moves.pop();
    return *this;
}

std::vector<Move> Board::getMoves() {
    torch::Tensor predicate = data == 0;
    torch::Tensor free = torch::nonzero(predicate[0] & predicate[1]);
    std::vector<Move> moves;
    for (int i = 0; i < free.size(0); i++) {
        Move move;
        move.row = free[i][0].item<int>();
        move.col = free[i][1].item<int>();
        move.player = turn;
        moves.push_back(move);
    }
    return moves;
}

torch::Tensor& Board::getData() {
    return data;
}

Result Board::getResult() {
    for (int p = 0; p < 2; p++) {
        bool diag1 = ((data[p][0][0] == 1) & (data[p][1][1] == 1) & (data[p][2][2] == 1)).item<bool>();
        bool diag2 = ((data[p][2][0] == 1) & (data[p][1][1] == 1) & (data[p][0][2] == 1)).item<bool>();
        for (int i = 0; i < 3; i++) {
            bool row = torch::all(data.index({p, i, torch::indexing::Slice()}) == 1).item<bool>();
            bool col = torch::all(data.index({p, torch::indexing::Slice(), i}) == 1).item<bool>();

            if (row || col || diag1 || diag2) {
                return (Result) p;
            }
        }
    }
    if (torch::all((data[0] == 1) | (data[1] == 1)).item<bool>()) {
        return Result::STALEMATE;
    }
    return Result::NONE;
}

Player Board::getTurn() {
    return turn;
}
