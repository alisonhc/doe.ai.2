from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from shutil import copyfileobj, move, rmtree
from tempfile import SpooledTemporaryFile
from re import sub, findall

_dataDir = Path("data")
if not _dataDir.exists():
    _dataDir.mkdir()


class _AbstractDataSource(ABC):
    @property
    @abstractmethod
    def _url(self):
        return ""

    @property
    @abstractmethod
    def _localPath(self):
        return ""

    @abstractmethod
    def getData(self):
        pass


class _AliceText(_AbstractDataSource):
    @property
    def _localPath(self):
        return _dataDir / "alice_text"

    @property
    def _url(self):
        return "https://ia801405.us.archive.org/18/items/alicesadventures19033gut/19033.txt"

    def getData(self, force=False):
        if not self._localPath.exists() or force:
            with urlopen(self._url) as response:
                data = response.read().decode("ascii")
            data = data.split('\r\n')[76:-371]  # this is just hardcoded to remove extraneous lines from the data
            data = [line for line in data if line != "" and "*" not in line and "Illustration" not in line]
            data = sub(r"[\",'();:]", "", " ".join(data))  # remove punctuation except ., !, and ?
            data = sub(r"--", " ", data)  # turn dashes into spaces
            data = sub(r"_([\w]+)?_", r"\1", data)  # eliminates any underscores around a word
            data = sub(r"\s+", " ", data)  # make all whitespace just a single space
            data = data.lower()
            data = sub(r"([!?.+])+\s*", r" \1\n", data)  # put spaces before punctuation and newlines after them
            with open(self._localPath, 'w') as f:
                f.write(data)

        with open(self._localPath, 'r') as f:
            data = [line.split() for line in f]
        return data


class _CornellMovieCorpus(_AbstractDataSource):
    @property
    def _localPath(self):
        return _dataDir / "cornell_movie_corpus"

    @property
    def _url(self):
        return "http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip"

    def getData(self, writeToFiles=True):
        if not self._localPath.exists():
            rootZipDir = "cornell movie-dialogs corpus"
            with urlopen(self._url) as response, SpooledTemporaryFile() as tmp:
                copyfileobj(response, tmp)
                with ZipFile(tmp) as zipTmp:
                    infoList = zipTmp.infolist()
                    for info in infoList:
                        pathFile = Path(info.filename)
                        if pathFile.parts[0] == rootZipDir and pathFile.stem != ".DS_Store":
                            zipTmp.extract(info, self._localPath)
            for p in self._localPath.joinpath(rootZipDir).iterdir():
                move(str(p), str(self._localPath))
            rmtree(str(self._localPath / rootZipDir))

        with open(self._localPath / "movie_conversations.txt", "r") as f:
            conversations = [findall(r"\d+", line.split("+++$+++")[-1]) for line in f]

        # for some reason, python can't parse the file at all without errors="ignore", something about uft-8 encoding
        with open(self._localPath / "movie_lines.txt", "r", errors="ignore") as f:
            lineDict = {}
            for line in f:
                _line = line.split("+++$+++")
                id = _line[0][1:-1]
                lineDict[id] = _line[-1].strip() # get rid of newlines and beginning spaces

        for i, conversation in enumerate(conversations):
            conversations[i] = [lineDict[lineId] for lineId in conversation]

        if writeToFiles:
            conversationPath = self._localPath / "input_output"
            if not conversationPath.exists():
                conversationPath.mkdir()
            inputFile = conversationPath / "inputs.txt"
            outputFile = conversationPath / "outputs.txt"
            inputs = [c[:-1] for c in conversations if len(c) >= 2]
            outputs = [c[1:] for c in conversations if len(c) >= 2]
            with open(inputFile, "w") as f:
                f.write("\n".join([line for c in inputs for line in c]))
            with open(outputFile, "w") as f:
                f.write("\n".join([line for c in outputs for line in c]))

            return conversations, inputFile, outputFile
        else:
            return conversations, None, None


class DataSource(Enum):
    ALICE_TEXT = _AliceText()
    CORNELL_MOVIE_CORPUS = _CornellMovieCorpus()
    CORNELL_MOVIE_QUOTES_CORPUS = "https://www.cs.cornell.edu/~cristian/memorability_files/cornell_movie_quotes_corpus.zip"
