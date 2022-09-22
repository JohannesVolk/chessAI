import os
import pathlib
import requests
import bz2
import zipfile
import io

def download_games_from_url(url = "https://database.lichess.org/standard/lichess_db_standard_rated_2015-04.pgn.bz2"):
    response = requests.get(url)

    with open("./chessai/data/pgn/lichess.pgn", "wb+") as file:
        file.write(bz2.decompress(response.content))

def download_zip(url : str, name : str, path : pathlib.Path):
    if path.exists():
        print(f"{name} already exists")
        return
    else:
        path.mkdir(parents=True)
        
    print(f"start downloading {name}")
    
    response = requests.get(url).content
    
    with zipfile.ZipFile(io.BytesIO(response)) as zip_ref:
        zip_ref.extractall(path)
        
    print(f"finished downloading {name}")

def dowload_polyglot_from_url(url = "https://maughancdn.s3.amazonaws.com/chess/The%20Baron/baronbook30.zip"):
    # further books#
    # https://digilander.libero.it/taioscacchi/archivio/Titans.zip
    # https://digilander.libero.it/taioscacchi/archivio/Human-polyglot.zip
    
    path = pathlib.Path("./data/polyglot")
    download_zip(url, "polyglot", path)

def download_syzygy_from_url(url = "https://bicjia.dm.files.1drv.com/y4meGhfz4XMHtA4HTvh-EGh_83H1W7RPStJCgo2DuO1iVulrttPMSvgNJF1kxJ7iA5c3aIpEQBqDTPVjyBglF1hVo3qzG875HyveHYMylIn8STsHAPSzhYE1Vk2BNOO_rQeV2Bb5e-uJrvUormbRn_tj0YPOtBa8cbOm8jYkH7vpjR9iQ-LwykaoPg3fLlaQCA1oG4Qja8TWyNGUorgMFmu_A"):
    # linked from https://chess.massimilianogoi.com/download/tablebases/
    
    path = pathlib.Path("./data/syzygy")
    download_zip(url, "syzygy", path)
    
    
def download_stockfish_from_url(url = "https://stockfishchess.org/files/stockfish_15_linux_x64_ssse.zip"):
    # linked from https://stockfishchess.org/download/linux/
    # you migh need to change the version for your OS / Hardware
    
    path = pathlib.Path("./data/stockfish")
    name = "stockfish_15_linux_x64_ssse"
    download_zip(url, name, path)
    
    # make the stockfish binary executable
    os.system(f"chmod 777 data/stockfish/{name}/stockfish_15_x64_ssse")
