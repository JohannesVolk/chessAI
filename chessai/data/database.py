import itertools
from pathlib import Path
import sqlite3
from typing import Iterator, Tuple
import chess
import chess.pgn
import chess.engine

from tqdm import tqdm
import hashlib

def normalize(evaluation):
    # calulated values from previous runs on the dataset
    evaluation = max(min(evaluation, 2500), -2500)
    mean_data = 26.4490
    variance_data = 647.956
    max_data = 3.8175
    min_data = -3.8992
    evaluation = (evaluation - mean_data) / variance_data
    evaluation = (max_data - evaluation) / (max_data - min_data)
    return evaluation

def game_2_board_properties(game: chess.pgn.Game) -> Iterator[chess.Board]:

    game = game.next()
    if game is None: return
    current_board = game.board()
    score = game.eval()

    if score is None:
        return

    while (game := game.next()) is not None:
        # game is now the next gamenode (representing the board after the next move)

        # we have endgame / opening tables for this
        if len(current_board.piece_map()) <= 5 or current_board.fullmove_number < 6:
            current_board = game.board()
            score = game.eval()
            if score is None:
                return
            continue

        if current_board.turn == chess.BLACK:
            current_board.apply_mirror()

        # get wdl expect for current player (always mirrored to be white player)
        evaluation = score.white().score()
        if evaluation is None: return
                
        # if evaluation.is_mate():
        #     mate_in = evaluation.mate()
        # else:
        #     score = evaluation.score()

        # normalize evaluation
        evaluation = normalize(evaluation)
        
        yield current_board, game.move, evaluation
        # play the next move
        current_board = game.board()
        score = game.eval()
        if score is None:
            return

    evaluation = score.white().score()
   
    if evaluation is None: return
    evaluation = normalize(evaluation)

    # return the last board (mate position) -> therefore None as next move
    yield current_board, None, evaluation


def encode_board(board: chess.Board):
    def coord_gen():
        # could also use but this causes overhead due to packing and unpacking the items
        for square, piece in board.piece_map().items():
            yield (piece.piece_type - 1) + (int(piece.color) * 6)
            yield square // 8
            yield square % 8

    return bytearray(coord_gen())


def games_generator(path, num_games) -> Iterator[chess.pgn.Game]:
    pgn = open(path)
    i = 0
    while True:
        if (game := chess.pgn.read_game(pgn)) is not None:
            yield game
            i += 1
            if i == num_games:
                return
        else:
            return


class Database:

    cur: sqlite3.Cursor
    con: sqlite3.Connection

    def add_positions(self, positions_with_evals):
        self.con.executemany(
            "INSERT or IGNORE INTO positions VALUES(?, ?, ?)", positions_with_evals
        )

    def get_positions(self, count):
        return self.con.execute(
            "SELECT * FROM positions LIMIT (?)", (count,)
        ).fetchall()

    def get_position(self, index):
        return self.con.executemany(
            "SELECT * FROM positions LIMIT 1 OFFSET (?)", (index,)
        ).fetchall()

    def store_positions_from_pgn_file(self, path, num_games):

        game_count_approx = (
            Path(path).stat().st_size // Path("./data/pgn/reference.pgn").stat().st_size
        )

        print(f"approximately {game_count_approx} games in pgn at {path}")
        games = tqdm(games_generator(path, num_games), total=num_games)

        # only add games that haven't been stored yet

        def not_in_database(game: chess.pgn.Game):
            hash_ = hashlib.sha1(repr(game.headers).encode("utf-8")).digest()
            self.cur.execute("SELECT * FROM games WHERE hash = (?)", (hash_,))
            not_in_db = self.cur.fetchone() is None
            if not_in_db:
                self.cur.execute("INSERT INTO games VALUES(?)", (hash_,))
            return not_in_db

        games = filter(not_in_database, games)

        # now 1 iterator instead of a game-iterator that yields board-iterators
        board_properties = itertools.chain.from_iterable(map(game_2_board_properties, games))

        def process_board(board_property: Tuple[chess.Board, chess.Move, float]):
            board, next_move, evaluation = board_property
            encoded_board = encode_board(board)
            if next_move is None:
                # just assign "no next move" to 64**2 as this move doesn't exist anyway
                move_id = 64 ** 2 - 1
            else:
                move_id = next_move.from_square + 64 * next_move.to_square

            return encoded_board, move_id, evaluation

        self.add_positions(map(process_board, board_properties))

    def __enter__(self):
        self.con = sqlite3.connect("data/database.db")
        self.cur = self.con.cursor()
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS positions
                    (data BLOB NOT NULL, move_id INTEGER NOT NULL, evaluation real NOT NULL, UNIQUE(data))"""
        )
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS games
                    (hash text, UNIQUE(hash))"""
        )
        return self

    def __exit__(self, *_):
        self.con.commit()
        self.con.close()
