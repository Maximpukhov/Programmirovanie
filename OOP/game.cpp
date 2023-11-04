#include <iostream>
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>

// Абстрактный класс Game
class Game {
public:
    virtual void play() = 0; // Абстрактный метод для начала игры
    virtual ~Game() {}
};

// Класс Board для представления игрового поля
class Board {
private:
    std::vector<std::vector<char>> board;
    const int size;

public:
    Board(int size, char initialSymbol) : size(size), board(size, std::vector<char>(size, initialSymbol)) {}
    Board(int size) : size(size), board(size, std::vector<char>(size, ' ')) {}

    void printBoard() {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                std::cout << board[i][j];
                if (j < size - 1) std::cout << "|";
            }
            std::cout << std::endl;
            if (i < size - 1) std::cout << std::string(size * 2 - 1, '-') << std::endl;
        }
    }

    bool makeMove(int row, int col, char player) {
        if (row >= 0 && row < size && col >= 0 && col < size && board[row][col] == ' ') {
            board[row][col] = player;
            return true;
        }
        return false;
    }

    bool checkWin(char player) {
        // Проверяем строки, столбцы и диагонали
        for (int i = 0; i < size; ++i) {
            if (std::count(board[i].begin(), board[i].end(), player) == size) return true;
            int columnCount = 0;
            for (int j = 0; j < size; ++j) {
                if (board[j][i] == player) columnCount++;
            }
            if (columnCount == size) return true;
        }
        // Проверяем диагонали
        int diag1Count = 0, diag2Count = 0;
        for (int i = 0; i < size; ++i) {
            if (board[i][i] == player) diag1Count++;
            if (board[i][size - i - 1] == player) diag2Count++;
        }
        if (diag1Count == size || diag2Count == size) return true;

        return false;
    }

    bool isFull() {
        for (const auto& row : board) {
            if (std::find(row.begin(), row.end(), ' ') != row.end()) {
                return false;
            }
        }
        return true;
    }
};

// Класс Player для представления игрока
class Player {
private:
    char token;

public:
    Player() : token(' ') {}
    Player(char token) : token(token) {}

    char getToken() const {
        return token;
    }
};

// Класс TicTacToe для игры в крестики-нолики
class TicTacToe : public Game {
private:
    Board board;
    Player player1;
    Player player2;
    Player* currentPlayer;

public:
    TicTacToe(int boardSize, char player1Token, char player2Token) : board(boardSize, ' '), player1(player1Token), player2(player2Token), currentPlayer(&player1) {}
    TicTacToe() : board(3), player1('X'), player2('O'), currentPlayer(&player1) {}

    void play() override {
        int turn = 0;
        int x, y;
        bool win = false;

        while (!board.isFull() && !win) {
            board.printBoard();
            std::cout << "Игрок " << currentPlayer->getToken() << ", введите координаты (строка столбец): ";
            std::cin >> x >> y;

            if (!board.makeMove(x - 1, y - 1, currentPlayer->getToken())) {
                std::cout << "Неверный ход, попробуйте еще раз." << std::endl;
                continue;
            }

            win = board.checkWin(currentPlayer->getToken());
            if (!win) {
                turn++;
                currentPlayer = (turn % 2 == 0) ? &player1 : &player2;
            }
        }
        board.printBoard();
        if (win) {
            std::cout << "Игрок " << currentPlayer->getToken() << " победил!" << std::endl;
        }
        else {
            std::cout << "Ничья!" << std::endl;
        }
    }
};

int main() {

    TicTacToe game;
    game.play();
    return 0;
}
