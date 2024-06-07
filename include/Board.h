#include <torch/torch.h>
#include <stack>

enum class Player {
    cross,
    circle,
};

enum class Result {
    CROSS,
    CIRCLE,
    STALEMATE,
    NONE,
};

struct Move {
    int row, col;
    Player player;
};

class Board {
private:
    torch::Tensor data;  
    Player turn;    
    std::stack<Move> moves;
public:
    Board();
    Board& makeMove(Move move);
    Board& undo();
    std::vector<Move> getMoves();
    torch::Tensor& getData();
    Result getResult();
    Player getTurn();
};