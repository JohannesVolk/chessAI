from typing import List
from chessai import __version__
import chess
from chessai.engine import MyEngine
import chess.engine


def test_check_mate_in_1_problems():
    engine = MyEngine()
    stockfish = chess.engine.SimpleEngine.popen_uci(
        "stockfish_15_linux_x64_ssse/stockfish_15_x64_ssse"
    )
    try:
        with open("tests/data/fen/mate2.fen") as file:
            list_ = file.readlines()

        boards = map(chess.Board, filter(lambda line: "/" in line, list_))

        for board in boards:
            move = stockfish.play(board, chess.engine.Limit(time=2)).move
            board.push(move)
            move = stockfish.play(board, chess.engine.Limit(time=2)).move
            board.push(move)
            move = engine.play(board, depth=1)
            board.push(move)
            assert board.is_game_over()
    finally:
        stockfish.quit()


def test_check_mate_in_2_problems():
    engine = MyEngine()
    stockfish = chess.engine.SimpleEngine.popen_uci(
        "stockfish_15_linux_x64_ssse/stockfish_15_x64_ssse"
    )
    try:
        with open("tests/data/fen/mate2.fen") as file:
            list_ = file.readlines()

        boards = map(
            chess.Board,
            filter(lambda line: "vs" not in line and "." not in line, list_),
        )

        for board in boards:
            move = engine.play(board, depth=1)
            board.push(move)

            move = stockfish.play(board, chess.engine.Limit(time=2)).move
            board.push(move)
            move = engine.play(board, depth=3)
            board.push(move)
            assert board.is_game_over()
            break
    finally:
        stockfish.quit()


def test_version():
    assert __version__ == "0.1.0"
