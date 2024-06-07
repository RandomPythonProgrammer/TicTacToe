#include "Table.h"

Table::Table(std::string path) {
    sqlite3_open(path.c_str(), &db);
}

Table& Table::updateQ(torch::Tensor state, double value) {
    std::lock_guard lock(mutex);
    sqlite3_stmt* stmt;
    std::string cmd = "SELECT value, count FROM q_values WHERE state = ?";
    sqlite3_prepare(db, cmd.c_str(), -1, &stmt, nullptr);
    std::string data = serialize(state);
    sqlite3_bind_blob(stmt, 1, data.data(), data.size(), SQLITE_STATIC);
    int code = sqlite3_step(stmt);
    double storedValue = 0;
    int count = 0;
    bool exists = false;
    if (code == SQLITE_ROW) {
        exists = true;
        storedValue = sqlite3_column_double(stmt, 1);
        count = sqlite3_column_int(stmt, 2);
    }
    sqlite3_finalize(stmt);

    storedValue = (storedValue * count + value)/(count + 1);
    count++;

    if (exists) {
        cmd = "UPDATE q_values SET (value, count) = (?, ?) WHERE state = ?";
        sqlite3_prepare(db, cmd.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_double(stmt, 1, storedValue);
        sqlite3_bind_int(stmt, 2, count);
        sqlite3_bind_blob(stmt, 3, data.data(), data.size(), SQLITE_STATIC);
    } else {
        cmd = "INSERT INTO q_values(state, value, count) VALUES (?, ?, ?)";
        sqlite3_prepare(db, cmd.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_blob(stmt, 1, data.data(), data.size(), SQLITE_STATIC);
        sqlite3_bind_double(stmt, 2, storedValue);
        sqlite3_bind_int(stmt, 3, count);
    }
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return *this;
}

std::pair<torch::Tensor, torch::Tensor> Table::getDataset(int size) {
    std::lock_guard lock(mutex);
    sqlite3_stmt* stmt;
    std::string cmd = "SELECT state, value FROM q_values ORDER BY RANDOM() LIMIT ?";
    sqlite3_prepare(db, cmd.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, size);

    torch::Tensor y = torch::zeros({size});
    std::vector<torch::Tensor> xBuffer;

    int code;
    for (int i = 0;;i++) {
        code = sqlite3_step(stmt);
        if (code == SQLITE_ROW) {
            const void* tensorData = sqlite3_column_blob(stmt, 0);
            int dataSize = sqlite3_column_bytes(stmt, 0);
            xBuffer.push_back(deserialize(tensorData, dataSize));
            y[i] = sqlite3_column_double(stmt, 1);
        } else if (code == SQLITE_DONE) {
            break;
        }
    }

    std::cerr << xBuffer.size() << std::endl;
    
    torch::Tensor x = torch::stack(xBuffer);
    sqlite3_finalize(stmt);
    return std::pair<torch::Tensor, torch::Tensor>(x, y);
}

Table::~Table() {
    sqlite3_close(db);
}


std::string serialize(torch::Tensor tensor) {
    std::ostringstream stream;
    torch::save(tensor, stream);
    return stream.str();
}

torch::Tensor deserialize(const void* data, int size) {
    std::string str((char*) data, size);
    std::istringstream stream(str);
    torch::Tensor tensor;
    torch::load(tensor, stream);
    return tensor;
}