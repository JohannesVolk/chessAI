from io import BytesIO
from typing import List
from zipfile import ZipFile
import requests
import re
from tqdm import tqdm

URL = "http://www.pgnmentor.com"

response = requests.get(f"{URL}/files.html").text


urls = []


subpages: List[str] = re.findall(
    r'(?<=href=").{0,20}\.zip(?=" class="view">Download)', response
)

for subpage in tqdm(subpages):
    file = requests.get(f"{URL}/{subpage}")
    subpage = subpage.replace("/", "_").replace(".zip", ".pgn")
    zipfile = ZipFile(BytesIO(file.content))
    zipfile.extractall("./chessai/data/pgn")
