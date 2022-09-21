from math import inf
from typing import Iterable, Optional
from chessai.network import ChessBoardEvalNN
from chessai.data.dataset import encode_board
import chess
import chess.engine
import torch
import chess.syzygy
import chess.polyglot


def moves(board: chess.Board) -> Iterable[chess.Board]:
    for move in board.legal_moves:
        board.push(move)
        yield board
        board.pop()


class MyEngine:
    def __init__(self) -> None:
        self.model = ChessBoardEvalNN.load_model([])
        self.model.eval()
        self.tablebase: Optional[chess.syzygy.Tablebase] = None
        self.next_move = chess.Move.null()

    def init_tablebase(self):
        self.tablebase = chess.syzygy.open_tablebase("./data/syzygy")

    def evaluate(self, board: chess.Board, depth: int) -> float:
        if board.is_checkmate():
            val = 100
        elif board.is_stalemate():
            val = 0
        elif len(board.piece_map()) <= 5:
            if self.tablebase is None:
                self.init_tablebase()

            dtz = -self.tablebase.probe_dtz(board)
            if dtz == 0:
                val = 0
            else:
                # scale so that dtz 1 is good and a higher dtz is bad (scale all outside of the [-1, 1] as dtz are 100% correct)
                val = 100 / dtz
        else:
            val = -self.negamax(board, depth - 1)

        return val

    def play(self, board, depth):

        with chess.polyglot.open_reader("./data/polyglot/baron30.bin") as reader:
            try:
                return reader.choice(board).move
            except IndexError:
                pass
        self.negamax(board, depth, root=True)
        return self.next_move

    def negamax(self, board: chess.Board, depth: int, root=False):

        if depth == 0:
            return self.evaluate_boards(board).item()

        best_val = -inf

        for move in board.legal_moves:
            board.push(move)

            val = self.evaluate(board, depth)

            board.pop()

            if val > best_val:
                best_val = val
                if root:
                    self.next_move = move

        return best_val

    # def evaluate_boards(self, boards : Iterable[chess.Board]) -> torch.Tensor:
    def evaluate_boards(self, board: chess.Board) -> torch.Tensor:

        # boards = map(encode_board, boards)
        # encoded_boards = torch.stack(list(boards))
        # return self.model.forward(encoded_boards)

        return self.model.forward(encode_board(board)[None, :])
