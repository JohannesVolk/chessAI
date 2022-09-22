from copy import deepcopy
from more_itertools import unzip
import torch
from torch.utils.data import Dataset
from chessai.data.database import Database
import chess
import chess.pgn
import chess.engine


def encode_board(board: chess.Board):
    encoding = torch.zeros((12, 8, 8), dtype=torch.float32)
    # always view the board from white perspective
    board = deepcopy(board)
    if board.turn == chess.BLACK:
        board.apply_mirror()

    for field, piece in board.piece_map().items():
        encoding[
            (piece.piece_type - 1) + (int(piece.color) * 6), field // 8, field % 8
        ] = 1

    return encoding


class ChessPositionsDataset(Dataset):
    def __init__(self, num_positions=1_000_000):
        # connect to db
        self.db = Database().__enter__()

        self.num_positions = num_positions
        
        
    def __getitem__(self, index):


        blob, move_id, evaluation = self.db.cur.execute(
            "SELECT * FROM positions WHERE rowid = (?)", (index+1,)
        ).fetchone()

        def blob_2_tensor(blob):

            # load coordinates for the figures from byte blob (transpose as the sparse_coo_tensor wants this format)
            # unfortunately the blob has to be copied as it is not writable and torch demands this even if it's not neccessary for us
            # as this is used read-only
            indices = torch.frombuffer(bytearray(blob), dtype=torch.uint8).view((-1, 3)).T
            dense_3d_tensor = torch.sparse_coo_tensor(
                indices, torch.ones(indices.shape[1]), (12, 8, 8), dtype=torch.float32
            ).to_dense()

            return dense_3d_tensor

        def move_id_2_tensor(move_id):
            space_to_move = torch.zeros(64 ** 2)
            space_to_move[move_id] = 1
            return space_to_move

        encoded_board = blob_2_tensor(blob)
        next_move = move_id_2_tensor(move_id)
        return encoded_board, next_move, torch.tensor(evaluation, dtype=torch.float32)




    def get_all_evaluations(self):

        _, _, winning = unzip(self.db.get_positions(self.num_positions))        
        
        winning = torch.tensor(list(winning), dtype=torch.float32)
    
        # if the dataset is use for the first time one has to record mean and variance to normalize the data in the database
        # print(torch.mean(winning))
        # print(torch.var(winning))
    
        # winning =  (winning - torch.mean(winning)) / torch.var(winning)**(0.5)
        
        # print(torch.max(winning))
        # print(torch.min(winning))
        
        # winning = (torch.max(winning) - winning) / (torch.max(winning) - torch.min(winning))
        
        return winning

    def get_all_move_ids(self):

        _, move_ids, _ = unzip(self.db.get_positions(self.num_positions))
        return torch.tensor(list(move_ids), dtype=torch.float32)

    def __len__(self):
        return self.num_positions
