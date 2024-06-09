#include <sqlite3.h>
#include <unordered_map>
#include <torch/torch.h>
#include <mutex>

class Table {
private:
    sqlite3 *db;
    std::mutex mutex;
public:
    Table(std::string path);

    Table& updateQ(torch::Tensor& state, double value);
    Table& updateQ(std::vector<std::vector<torch::Tensor>>& states, std::vector<std::vector<double>>& values);
    double getQ(torch::Tensor& state);
    std::pair<torch::Tensor, torch::Tensor> getDataset(int size);
    ~Table();
};

std::string serialize(torch::Tensor tensor);
torch::Tensor deserialize(const void* data, int size);