import requests
import bz2


URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2015-04.pgn.bz2"

response = requests.get(URL)

with open("./chessai/data/pgn/lichess.pgn", "wb+") as file:
    file.write(bz2.decompress(response.content))
